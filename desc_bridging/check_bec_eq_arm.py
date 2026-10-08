"""How much of the eq arm's BEC gap does the equalizer itself account for?

``deepmd-cace-eq_sA`` is ``cace-sea-lr``'s graph with ``ChargeEqLatent`` in place of
the plain ``EwaldPotential``, trained through ``deepmd_cace`` on cace's own schedule.
Scored on the ``BEC/`` VASP set it lands ~15% above ``cace-sea-lr`` on pooled RMSE.

This script asks whether that gap is something the equalizer *does*, by scoring the
same trained weights twice on the same frames:

* as trained - ``charge_eq_latent.enabled: true``, so ``q_eq`` enters ``Polarization``;
* with ``charge_eq_latent`` removed from the config - the plain kernel, so the head's
  raw ``q`` enters it.

``ChargeEqLatent`` holds no parameters or buffers, so the checkpoint's state dict loads
strictly into both. The pair differs in exactly one thing, and because the frames are
the same the two columns can be subtracted element by element.

This is a decomposition of the trained arm, not a pair of trained arms: the weights
were fit *with* the equalizer in the graph, so the "off" column is a counterfactual at
those weights, not the model a plain-kernel run would have reached.

Usage:
    python check_bec_eq_arm.py --frames 100
"""
import argparse
import copy
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("DEEPMD_CACE_DEVICE", "cpu")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check_bec as CB  # noqa: E402
import check_bec_arm as CBA  # noqa: E402
from eval_vasp_bec import ATOMIC_NUMBER, EPS_INF, read_outcar_bec, read_poscar  # noqa: E402

DEFAULT_CKPT = os.path.join(
    HERE, "..", "DeePMD-kit-FastLearn", "campaign_fastlearn", "deepmd", "runs",
    "deepmd-cace-eq_sA", "best_model.pth",
)
VASP_DIR = "/root/app/deepmd-les/BEC"
RUNS = os.path.join(HERE, "..", "DeePMD-kit-FastLearn", "campaign_fastlearn",
                    "deepmd", "runs")


def vasp_frames(nframes, vasp_dir=VASP_DIR):
    indices = sorted(
        int(f.split(".")[1]) for f in os.listdir(os.path.join(vasp_dir, "poscars"))
    )[:nframes]
    frames = []
    for idx in indices:
        coord, cell, elem = read_poscar(os.path.join(vasp_dir, "poscars", f"POSCAR.{idx}"))
        ref = read_outcar_bec(os.path.join(vasp_dir, "outcars", f"OUTCAR.{idx}"))
        frames.append(dict(coord=coord, cell=cell, elem=elem, ref=ref))
    return frames


def load_with_eq(path, enabled):
    """The checkpoint's weights in the architecture with the equalizer on or off."""
    from deepmd.pt.model.descriptor.se_a import DescrptSeA
    from deepmd_cace.model import build_model

    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(blob["config"])
    long_range = config["model"]["long_range"]
    if enabled:
        # the checkpoint was trained with it, so its own config already asks for it
        long_range.setdefault("charge_eq_latent", {"enabled": True})
    else:
        long_range["charge_eq_latent"] = {"enabled": False}
    descriptor = DescrptSeA(
        **{k: v for k, v in config["model"]["descriptor"].items() if k != "type"}
    )
    model = build_model(config["model"], descriptor, "cpu")
    model.load_state_dict(blob["state_dict"], strict=True)
    return model.eval()


def score(model, frames, scale):
    head = CB.charge_head_of(model)
    dtype = CB.model_dtype(model)
    pred, dft, elem = [], [], []
    q_net = []
    for fr in frames:
        coordinates = fr["coord"]
        numbers = torch.tensor([ATOMIC_NUMBER[e] for e in fr["elem"]])
        pos = torch.tensor(coordinates, dtype=dtype, requires_grad=True)
        cell = torch.tensor(fr["cell"], dtype=dtype).reshape(1, 3, 3)
        batch = torch.zeros(len(fr["elem"]), dtype=torch.int64)
        _, q_kernel = CBA.kernel_charges(model, head, pos, cell, batch, numbers)
        bec = CB.born_charges(pos, cell, batch, q_kernel, True)
        pred.append(np.diagonal(bec, axis1=1, axis2=2))
        dft.append(np.diagonal(fr["ref"], axis1=1, axis2=2))
        elem.append(fr["elem"])
        q_net.append(float(q_kernel.detach().sum()) * CB.CHARGE_UNIT)
    signed = np.concatenate([p.ravel() for p in pred])
    dft = np.concatenate([d.ravel() for d in dft])
    elem = np.concatenate([np.repeat(e, 3) for e in elem])
    sign = -1.0 if signed[elem == "O"].mean() > 0 else 1.0
    return dict(pred=signed * sign * scale, raw=signed * sign, dft=dft, elem=elem,
                sign=sign, q_net=np.array(q_net), per_frame=pred)


def metrics(pred, dft, elem):
    out = {}
    for name in ("all", "O", "H"):
        m = np.ones_like(dft, dtype=bool) if name == "all" else (elem == name)
        residual = pred[m] - dft[m]
        out[name] = dict(
            rmse=float(np.sqrt(np.mean(residual ** 2))),
            r2=float(1.0 - np.sum(residual ** 2) / np.sum((dft[m] - dft[m].mean()) ** 2)),
            bias=float(residual.mean()),
            mean=float(pred[m].mean()),
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--lr-weight", type=float, default=None,
                        help="weight on E_LR; default: the checkpoint's own config")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    if args.lr_weight is None:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        long_range = blob["config"]["model"].get("long_range", {})
        weight = (float(long_range.get("weight", 1.0))
                  if long_range.get("combine_potentials", False) else 1.0)
    else:
        weight = args.lr_weight
    scale = float(np.sqrt(weight * EPS_INF))

    frames = vasp_frames(args.frames)
    label = args.label or os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    print(f"arm        : {label}")
    print(f"frames     : {len(frames)}")
    print(f"lr weight  : {weight:g}   scale sqrt(w*eps_inf) = {scale:.4f}")

    columns = {}
    for enabled in (True, False):
        model = load_with_eq(args.ckpt, enabled)
        name = "as trained (q_eq)" if enabled else "equalizer off (q)"
        columns[name] = score(model, frames, scale)
        result = columns[name]
        m = metrics(result["pred"], result["dft"], result["elem"])
        print(f"\n{name}:")
        print(f"  Z*_O = {result['pred'][result['elem'] == 'O'].mean():+.3f}   "
              f"Z*_H = {result['pred'][result['elem'] == 'H'].mean():+.3f}   "
              f"sign x{result['sign']:+.0f}   net q/frame {result['q_net'].mean():+.4f} e")
        print("  %-6s %9s %9s %9s %9s" % ("", "RMSE", "R^2", "bias", "mean"))
        for key in ("all", "O", "H"):
            v = m[key]
            print("  %-6s %9.4f %9.4f %+9.4f %9.4f"
                  % (key, v["rmse"], v["r2"], v["bias"], v["mean"]))

    left, right = columns["as trained (q_eq)"], columns["equalizer off (q)"]
    delta = left["pred"] - right["pred"]
    print("\nPaired difference, as trained minus equalizer off (same weights, same frames):")
    for name in ("all", "O", "H"):
        m = np.ones_like(delta, dtype=bool) if name == "all" else (left["elem"] == name)
        print("  %-6s mean %+.5f e   mean|dZ*| %.5f e   max|dZ*| %.5f e   "
              "rms(dZ*)/rms(Z*) %.4f"
              % (name, delta[m].mean(), np.abs(delta[m]).mean(), np.abs(delta[m]).max(),
                 np.sqrt(np.mean(delta[m] ** 2)) / np.sqrt(np.mean(left["pred"][m] ** 2))))
    print(f"  charge drift removed by the equalizer: {right['q_net'].mean():+.4f} -> "
          f"{left['q_net'].mean():+.4f} e per frame")

    arm_dir = os.path.dirname(os.path.abspath(args.ckpt))
    if os.path.basename(arm_dir).startswith("deepmd-cace-"):
        np.savez(
            os.path.join(HERE, "bec_eq_arm_compare.npz"),
            trained_pred=left["pred"], trained_dft=left["dft"], elem=left["elem"],
            off_pred=right["pred"], scale=scale, lr_weight=weight,
            frames=len(frames), arm=os.path.basename(arm_dir),
        )
        print("\nwrote bec_eq_arm_compare.npz")


if __name__ == "__main__":
    main()
