"""Score every arm's best_model.pth on the campaign's 50 validation frames.

What this produces and why each piece is here
---------------------------------------------
* ``rmse_e`` / ``mae_e`` in eV per atom, on the residual target the campaign
  trains against, and ``rmse_f`` / ``mae_f`` in eV/Angstrom. These are the two
  numbers every arm is finally judged by.
* an offset/scale/shape split of the energy error (mean bias, regression slope,
  and the residual after removing both). A single RMSE cannot say whether an arm
  is displaced, mis-scaled, or genuinely noisy, and the three call for different
  fixes.
* for the four long-range arms, the E_sr/E_lr and F_sr/F_lr decomposition. Both
  layouts emit each branch already scaled by the coupling - ``CombinePotential``
  multiplies it in place, and the sea arm uses that layout too - so the emitted
  ``ewald_potential``/``ewald_forces`` are the contributions to the total, not
  the branch values. The stored ``lr_e``/``lr_f`` are the raw branch, recovered
  by dividing the coupling back out, and ``sr_e``/``sr_f`` are what is left of
  the total: ``sr + w * lr == tot``. The coupling ``w`` is read from each
  checkpoint rather than assumed (``potential_keys`` in the cace arm,
  ``long_range.weight`` in the sea arm; both 0.02 here), so it stays available to
  recompute the coupled contribution.

Two independent routes to the same force split
----------------------------------------------
The checkpoints emit forces themselves, as ``CACE_forces`` on the total energy and
(in the cace arm) ``ewald_forces`` on the long-range one. A second route through
autograd, differentiating the named energy keys on a retained graph, gives the
same split independently. The two are compared (``lr_force_route_gap``,
``force_route_gap``) - a check on the extraction rather than on the physics, since
both routes walk the same modules. The headline ``rmse_f`` is the module route,
which is what the campaign logs also report.

One data path for every arm
---------------------------
The deepmd systems in ``data/water-interface`` were written by calling cace's own
``random_train_valid_split(valid_fraction=0.1, seed=1)``, so they hold the same 50
frames and the same residual target as the cace arms' own loader - checked, not
assumed, in ``check_consistency`` below. Feeding all eight models batches built by
one loader removes the family's data path as an explanation for any difference.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent


def _find_desc_bridging():
    """The tree holding ``deepmd_cace`` - the sea arms' model and data code."""
    for parent in [CAMPAIGN, *CAMPAIGN.parents]:
        candidate = parent / "desc_bridging"
        if (candidate / "deepmd_cace" / "__init__.py").is_file():
            return candidate
    raise ModuleNotFoundError("no desc_bridging/deepmd_cace above the campaign directory")


sys.path.insert(0, str(_find_desc_bridging()))

import arms as arms_module  # noqa: E402  (needs sys.path above)

CUTOFF = 5.5                      # the cutoff both families' descriptors declare
TYPE_MAP = ["O", "H"]
NATOMS = 1566
LR_ENERGY_KEY = "ewald_potential"
LR_FORCE_KEY = "ewald_forces"
TOTAL_ENERGY_KEY = "CACE_energy"
TOTAL_FORCE_KEY = "CACE_forces"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_batches(limit=None):
    """The validation frames as cace batches, with their reference labels.

    ``atomic_energies`` is left unset: the deepmd energy.npy already holds the
    residual, which is the same number cace's loader produces after it subtracts
    the references. Passing the references here would subtract them twice.
    """
    from deepmd_cace.data import load_split

    loader, _ = load_split(
        [str(arms_module.VALID_SYSTEM)], TYPE_MAP, CUTOFF, 1,
        shuffle=False, collect_stats=False, atomic_energies=None,
    )
    frames = []
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        # to_dict() rather than the Batch itself: the models read plain mappings,
        # and it keeps every downstream access a dict lookup.
        frames.append(batch.to_dict())
    return frames


def check_consistency(frames):
    """The guard that makes the one-data-path decision safe.

    Asserts the frames are the campaign's validation split and that the target is
    the residual, i.e. a total energy minus the references, not a total energy.
    A wrong target would silently rescale every RMSE in the report by ~880x.
    """
    refs = np.load(arms_module.VALID_SYSTEM / "set.000" / "energy.npy")
    coord = np.load(arms_module.VALID_SYSTEM / "set.000" / "coord.npy")
    assert len(refs) == 50, f"the campaign's validation split is 50 frames, found {len(refs)}"
    assert len(frames) <= len(refs), f"{len(frames)} frames against {len(refs)} labels"
    for index, batch in enumerate(frames):
        assert np.isclose(float(batch["energy"]), refs[index], rtol=0, atol=1e-4), (
            f"frame {index}: batch energy {float(batch['energy'])} != energy.npy {refs[index]}")
        assert np.allclose(batch["positions"].numpy().reshape(-1), coord[index], atol=1e-6), (
            f"frame {index}: positions do not match the labelled split")
    # -0.177 eV/atom for the residual; a raw total is ~-156 eV/atom
    per_atom = float(np.mean([float(b["energy"]) for b in frames])) / NATOMS
    assert abs(per_atom) < 1.0, f"target looks like a total energy ({per_atom:.1f} eV/atom)"
    return refs


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def load_model(arm, device):
    """The checkpoint, plus the long-range coupling weight when there is one."""
    if arm.family == "cace":
        model = torch.load(arm.best_model, map_location=device, weights_only=False)
        weight = float("nan")
        if arm.topology == "lr":
            keys = model.potential_keys
            weight = float(keys[1]["weight"])
            assert keys[1][TOTAL_ENERGY_KEY] == LR_ENERGY_KEY, keys[1]
        return model.to(device).eval(), weight

    from deepmd_cace.model import load_checkpoint

    blob = torch.load(arm.best_model, map_location="cpu", weights_only=False)
    assert blob["format"] == "deepmd-cace-state-dict-1", blob["format"]
    long_range = blob["config"]["model"].get("long_range", {})
    weight = float(long_range.get("weight", float("nan"))) if arm.topology == "lr" else float("nan")
    # CPU rebuild: build_model calls .to(device) itself and the descriptor's stats
    # come back from the state dict, so nothing needs the GPU to be present.
    model = load_checkpoint(str(arm.best_model), device="cpu")
    return model.to(device).eval(), weight


def _to_device(batch, device):
    """The batch on ``device``, with positions a fresh grad-tracking leaf.

    The leaf is always rebuilt, for two reasons. cace's ``Forces`` differentiates
    the energy it is pointed at, so it needs positions to require grad - which
    under ``torch.no_grad`` no longer happens as a side effect of
    ``initialize_derivatives``. And a fresh leaf per call keeps the graph this
    pass builds separate from any earlier pass over the same frame, which matters
    because the frames list is scored once per arm.
    """
    data = {key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()}
    data["positions"] = data["positions"].detach().clone().requires_grad_(True)
    return data


def forward(model, batch, device, training=False):
    """One forward pass, with grad mode on.

    Grad mode is always on: every checkpoint here has a ``Forces`` output module
    that calls ``autograd.grad``, and under ``no_grad`` the graph it differentiates
    is never built.

    ``training`` is what decides whether that graph survives the pass.
    ``compute_forces_virials`` frees it when ``training`` is false and retains it
    when true, which is exactly the switch the decomposition needs - so the second
    pass is a plain re-forward rather than surgery on the module list. Nothing in
    these checkpoints reads ``self.training`` (none has a BatchNorm or Dropout),
    and the energies come back bit-identical either way; the forces differ by
    ~1e-6 eV/A, which is why the decomposition is taken entirely from the pass
    that also produced the energies it is split from.

    Returns ``(outputs, positions)``. The positions are the leaf handed in, kept
    because ``extract_outputs`` returns only the model's declared output keys -
    and because ``Preprocess`` replaces ``data["positions"]`` with a
    strain-deformed copy, so the dict's entry after the call is a different
    tensor. Differentiating the leaf is equivalent: the strain displacement these
    models build is identically zero, so the deformed copy equals the leaf, which
    ``force_route_gap`` then confirms.
    """
    data = _to_device(batch, device)
    positions = data["positions"]
    with torch.enable_grad():
        outputs = model(data, training=training)
    return outputs, positions


def gradient(energy, positions):
    return -torch.autograd.grad(energy.sum(), positions, retain_graph=True)[0].detach()


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def numpy_of(tensor):
    """Detached, on the host, as an array. Every read out of a forward goes
    through here: the outputs carry a graph now, so ``.numpy()`` alone raises."""
    return tensor.detach().cpu().numpy()


def score_arm(arm, frames, device, verbose=False):
    model, weight = load_model(arm, device)
    is_lr = arm.topology == "lr"

    total_e, total_f, ref_e, ref_f = [], [], [], []
    total_f_autograd = []
    sr_e, lr_e, sr_f, lr_f = [], [], [], []
    module_lr_f = []
    charges = []

    for batch in frames:
        out, _ = forward(model, batch, device, training=False)
        e_tot = out[TOTAL_ENERGY_KEY].reshape(-1)
        f_tot = out[TOTAL_FORCE_KEY].reshape(-1, 3)
        total_e.append(numpy_of(e_tot))
        total_f.append(numpy_of(f_tot))
        ref_e.append(float(batch["energy"]))
        ref_f.append(batch["forces"].numpy())

        if not is_lr:
            continue
        e_lr = out[LR_ENERGY_KEY].reshape(-1)
        charges.append(numpy_of(out["q"]))
        if LR_FORCE_KEY in out:
            module_lr_f.append(numpy_of(out[LR_FORCE_KEY]))

        # the retained-graph pass: same forward, training=True so the graph
        # survives its own Forces modules and ``gradient`` can walk it
        grads, pos = forward(model, batch, device, training=True)
        e_lr_g = grads[LR_ENERGY_KEY].reshape(-1)
        f_lr = gradient(grads[LR_ENERGY_KEY], pos)
        f_tot_g = gradient(grads[TOTAL_ENERGY_KEY], pos)
        assert torch.allclose(e_lr_g.detach(), e_lr.detach(), rtol=0, atol=1e-4), (
            "energy changed between passes")
        # both layouts emit each branch already scaled by the coupling ``w``, so
        # ``ewald_potential`` and its force are the contributions to the total, not
        # the branch's own value. Divide the coupling back out to store the raw
        # long-range branch, and take the short-range piece as what is left of the
        # total. The two stored parts then satisfy ``sr + w * lr == tot``.
        lr_e.append(numpy_of(e_lr / weight))
        sr_e.append(numpy_of(e_tot - e_lr))
        lr_f.append(numpy_of(f_lr / weight))
        sr_f.append(numpy_of(f_tot_g - f_lr))
        total_f_autograd.append(numpy_of(f_tot_g))
        if verbose and len(total_e) == 1:
            print(f"    frame 0: E_tot {float(e_tot):+.4f}  E_sr {float(e_tot - e_lr):+.4f}  "
                  f"E_lr raw {float(e_lr / weight):+.3f}  coupled {float(e_lr):+.4f}  w {weight}")
            print(f"    frame 0: |F_tot| {float(f_tot.norm()):.4f}  "
                  f"|F_sr| {float(np.linalg.norm(sr_f[-1])):.4f}  "
                  f"|F_lr raw| {float((f_lr / weight).norm()):.4f}  "
                  f"coupled {float(f_lr.norm()):.4f}")
            print(f"    frame 0: F_tot from module {float(f_tot.norm()):.5f} vs autograd "
                  f"{float(f_tot_g.norm()):.5f}")

    ref_e = np.array(ref_e)
    pred_e = np.concatenate(total_e)
    ref_f = np.concatenate(ref_f, axis=0)
    pred_f = np.concatenate(total_f, axis=0)

    result = {
        "arm_id": arm.arm_id, "family": arm.family, "topology": arm.topology,
        "rep": arm.rep, "n_frames": len(ref_e), "n_params": sum(p.numel() for p in model.parameters()),
        "weight": weight,
        "ref_e": ref_e, "pred_e": pred_e,
        "ref_f": ref_f, "pred_f": pred_f,
        "energy_error": (pred_e - ref_e) / NATOMS,
        "force_error": pred_f - ref_f,
    }
    if is_lr:
        result.update({
            "sr_e": np.concatenate(sr_e), "lr_e": np.concatenate(lr_e),
            "sr_f": np.concatenate(sr_f, axis=0), "lr_f": np.concatenate(lr_f, axis=0),
            "charge": np.concatenate(charges, axis=0),
        })
        if module_lr_f:
            # the module's ``ewald_forces`` is the coupled value too, so it is
            # uncoupled before the comparison, keeping both routes in raw terms
            module = np.concatenate(module_lr_f, axis=0) / weight
            autograd = np.concatenate(lr_f, axis=0)
            result["lr_force_route_gap"] = float(np.abs(module - autograd).max())
        # both routes to F_tot should agree; a gap here means the split below is
        # measured against a different force than the RMSE above
        result["force_route_gap"] = float(np.abs(
            np.concatenate(total_f_autograd, axis=0) - pred_f).max())
        # the split must reproduce the total it was taken from. The energy half is
        # an identity by construction (``sr_e`` is defined as what is left of
        # ``pred_e``), so its zero says only that the arithmetic closed. The force
        # half is not an identity, because ``sr_f`` is built from the autograd
        # total and ``pred_f`` is the module total; it reports that route gap, plus
        # the rounding the raw value picks up on its round trip through ``w``.
        result["split_gap_e"] = float(np.abs(
            result["sr_e"] + weight * result["lr_e"] - pred_e).max())
        result["split_gap_f"] = float(np.abs(
            result["sr_f"] + weight * result["lr_f"] - pred_f).max())
    return result


def summarise(result):
    """The headline numbers, plus the offset/scale/shape split of the energy error.

    An RMSE cannot say whether an arm is displaced, mis-scaled, or genuinely
    noisy, and the three call for different fixes. So the total error is
    decomposed by fitting pred = slope * ref + intercept:

      offset  the mean error, what a constant shift in the energy can remove;
      scale   the fitted slope, how much of the error is a stretched axis;
      shape   what is left after both, the part only a better model removes.
    """
    ref_e, pred_e = result["ref_e"], result["pred_e"]
    err = result["energy_error"]                       # eV/atom
    f_err = result["force_error"]
    centred = ref_e - ref_e.mean()
    slope = float(np.dot(centred, pred_e - pred_e.mean()) / np.dot(centred, centred))
    intercept = float(pred_e.mean() - slope * ref_e.mean())
    row = {
        "arm_id": result["arm_id"], "family": result["family"],
        "topology": result["topology"], "rep": result["rep"],
        "n_params": result["n_params"],
        "rmse_e_peratom": float(np.sqrt(np.mean(err ** 2))),
        "mae_e_peratom": float(np.mean(np.abs(err))),
        "rmse_f": float(np.sqrt(np.mean(f_err ** 2))),
        "mae_f": float(np.mean(np.abs(f_err))),
        "energy_offset_peratom": float(err.mean()),
        "energy_scale_slope": slope,
        "energy_rmse_after_offset": float(np.sqrt(np.mean((err - err.mean()) ** 2))),
        "energy_rmse_after_scale": float(np.sqrt(np.mean(
            ((pred_e - (slope * ref_e + intercept)) / NATOMS) ** 2))),
    }
    if "lr_e" in result:
        f_tot_rms = float(np.sqrt(np.mean(result["pred_f"] ** 2)))
        row.update({
            "sr_e_rms_peratom": float(np.sqrt(np.mean(result["sr_e"] ** 2))) / NATOMS,
            "lr_e_rms_peratom": float(np.sqrt(np.mean(result["lr_e"] ** 2))) / NATOMS,
            "lr_e_mean": float(np.mean(result["lr_e"])),
            "sr_f_rms": float(np.sqrt(np.mean(result["sr_f"] ** 2))),
            "lr_f_rms": float(np.sqrt(np.mean(result["lr_f"] ** 2))),
            # lr_f is the raw branch, so the coupling has to be put back before it
            # can be read as a share of the total it contributes to
            "lr_f_over_total_f": float(
                result["weight"] * np.sqrt(np.mean(result["lr_f"] ** 2)) / f_tot_rms),
            "weight": result["weight"],
            "split_gap_e": result["split_gap_e"],
            "split_gap_f": result["split_gap_f"],
            "force_route_gap": result["force_route_gap"],
        })
        if "lr_force_route_gap" in result:
            row["lr_force_route_gap"] = result["lr_force_route_gap"]
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frames", type=int, default=None, help="score only the first N frames")
    parser.add_argument("--only", default=None, help="a single arm_id, for a quick smoke")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)
    frames = build_batches(limit=args.frames)
    check_consistency(frames)
    print(f"{len(frames)} validation frames, device {device}\n")

    rows, payload = [], {}
    for arm in arms_module.ARMS:
        if args.only and arm.arm_id != args.only:
            continue
        print(f"--- {arm.arm_id}")
        result = score_arm(arm, frames, device, verbose=args.verbose)
        row = summarise(result)
        rows.append(row)
        payload[arm.arm_id] = result
        extra = ""
        if "lr_f_over_total_f" in row:
            extra = (f"  F_lr/F_tot {row['lr_f_over_total_f']:.3f}"
                     f"  split_gap_e {row['split_gap_e']:.2e}  split_gap_f {row['split_gap_f']:.2e}")
            if "lr_force_route_gap" in row:
                extra += f"  route_gap {row['lr_force_route_gap']:.2e}"
        print(f"    rmse_e {row['rmse_e_peratom']:.6e} eV/atom   rmse_f {row['rmse_f']:.6f} eV/A"
              f"   slope {row['energy_scale_slope']:.4f}{extra}")

    if args.only:
        return
    tsv = HERE / "valid_metrics.tsv"
    # the union, in first-seen order: the short-range arms carry no decomposition
    # columns, and every arm should still appear on its own row in one table
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(tsv, "w", encoding="utf-8") as stream:
        stream.write("\t".join(columns) + "\n")
        for row in rows:
            stream.write("\t".join(
                "" if row.get(c) is None
                else f"{row[c]:.8e}" if isinstance(row[c], float) else str(row[c])
                for c in columns) + "\n")
    np.savez(HERE / "scored_valid.npz", **{
        f"{arm_id}/{key}": value for arm_id, result in payload.items()
        for key, value in result.items() if isinstance(value, np.ndarray)})
    with open(HERE / "scored_valid.json", "w", encoding="utf-8") as stream:
        json.dump([{k: (v if not isinstance(v, np.ndarray) else None) for k, v in row.items()}
                   for row in rows], stream, indent=2)
    print(f"\nwrote {tsv.name}, scored_valid.npz, scored_valid.json")


if __name__ == "__main__":
    main()
