"""Lower-interface check for the hybrid_ener model: forward_lower.

`dp --pt freeze` / the C++ API / LAMMPS drive a model through `forward_lower`,
which takes the already-extended region (coordinates, types, neighbor list,
mapping) instead of a local frame. HybridLESModel overrides it so the LES/Ewald
long-range channel survives that path, and - because Ewald needs the cell
explicitly while the upstream lower signature has no cell - adds a trailing
optional `box` argument plus a `need_lower_box()` probe so the C++ side knows to
fill it.

`forward` is `forward_common` (which builds the extended region and delegates to
`forward_common_lower`) plus the same LES channel, so on the same extended inputs
the two entry points must agree. This script checks:

  1. forward vs forward_lower on energy / virial / force;
  2. that each route's per-atom virial sums to its own reported total;
  3. that both virials equal the finite-difference strain derivative
     -dE/deps (nine components, sheared cell), with the short-range-only virial
     as a negative control (it must fail, or the check has no power over the
     long-range channel);
  4. the symmetry sentinel: an antisymmetric part of the strain is a rigid
     rotation and leaves the energy invariant, so a symmetric tensor is
     mandatory. A virial that sums the long-range atomic term over local atoms
     only (dropping the ghosts, whose image positions carry the cell
     contribution) breaks this;
  5. that a missing cell raises instead of silently returning short-range-only
     numbers;
  6. that torch.jit.script keeps the method, its schema and its numerics.

Usage: python check_forward_lower.py [ckpt]
"""
import sys

import torch

from _common import build_model, build_water, load_checkpoint
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

EPS = 1e-4
I3 = torch.eye(3, dtype=torch.float64)
SHEAR = torch.tensor(
    [[1.0, 0.113, 0.0], [0.0, 1.0, 0.071], [0.047, 0.0, 1.0]], dtype=torch.float64
)
KEYS = ["energy", "virial", "extended_force", "extended_virial"]


def build_extended(model, coord, atype, box):
    """The extended region exactly as forward_common builds it internally.

    make_model.py hardcodes mixed_types=True there: at the lower interface the
    type distinction is the model's own job, not the neighbor list's.
    """
    ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
        coord, atype, model.get_rcut(), model.get_sel(), mixed_types=True, box=box
    )
    # the lower interface takes nlist before mapping
    return ext_coord, ext_atype, nlist, mapping


def call_lower(model, ext_coord, ext_atype, nlist, mapping, box):
    """forward_lower with every argument positional, so the same call works on a
    scripted module (whose signature is the exported schema, not the Python one)."""
    return model.forward_lower(
        ext_coord, ext_atype, nlist, mapping, None, None, True, None, box
    )


def reduce_ext(ext, nframes, nloc, mapping):
    """Sum extended per-atom quantities onto their local owners.

    Same reduction as communicate_extended_output (transform_output.py:239):
    scatter_reduce over dim 1, with the mapping expanded to the value's rank.
    """
    out = torch.zeros(
        [nframes, nloc] + list(ext.shape[2:]), dtype=ext.dtype, device=ext.device
    )
    # mapping is [nframes, nall]: expand the *source* to rank n, then scatter.
    idx = mapping.reshape(list(mapping.shape) + [1] * (ext.dim() - 2)).expand(list(ext.shape))
    return torch.scatter_reduce(out, 1, index=idx, src=ext, reduce="sum")


def strained(coord, box, shear, eps):
    """h -> h (I + eps), r -> r (I + eps) at fixed fractional coordinates."""
    nloc = coord.shape[1]
    box0 = box.reshape(3, 3) @ shear
    coord0 = coord.reshape(nloc, 3) @ shear
    frac = coord0 @ torch.linalg.inv(box0)
    b = box0 @ (I3.to(box0) + eps)
    return (frac @ b).reshape(1, nloc, 3), b.reshape(1, 9)


def main():
    coord, atype, box = build_water(device="cpu")
    model = (
        load_checkpoint(sys.argv[1], device="cpu")
        if len(sys.argv) > 1
        else build_model(device="cpu")
    )
    nframes, nloc = coord.shape[:2]
    dev = coord.device

    # ---------------------------------------------------------------- 1. agreement
    with torch.enable_grad():
        a = model(coord.clone(), atype.clone(), box.clone(), do_atomic_virial=True)
    ext_coord, ext_atype, nlist, mapping = build_extended(
        model, coord.clone(), atype.clone(), box.clone()
    )
    with torch.enable_grad():
        b = call_lower(model, ext_coord, ext_atype, nlist, mapping, box.clone())

    dw = float((ext_coord[:, :nloc] - coord).abs().max())
    print(f"{nframes} frames x {nloc} atoms   extended region {tuple(ext_coord.shape)}")
    print(f"|extended_coord[:, :nloc] - coord|_max = {dw:.3e}")
    print(f"keys: {sorted(b.keys())}")

    pairs = [
        ("energy", a["energy"], b["energy"]),
        ("virial", a["virial"], b["virial"]),
        ("force", a["force"], reduce_ext(b["extended_force"].unsqueeze(2), nframes, nloc, mapping)),
    ]
    worst = 0.0
    for name, ref, got in pairs:
        ref = ref.detach().double().reshape(nframes, -1)
        got = got.detach().double().reshape(nframes, -1)
        assert ref.shape == got.shape, (name, ref.shape, got.shape)
        d = float((ref - got).abs().max())
        scale = max(float(ref.abs().max()), 1e-30)
        worst = max(worst, d / scale)
        print(f"   {name:12s} bit-identical={bool(torch.equal(ref, got))}  "
              f"max|d|={d:.3e}  rel={d / scale:.2e}")

    # The two routes split the per-atom virial differently (forward sees the
    # caller's raw coordinates and spreads the cell term over the local atoms,
    # forward_lower sees the extended region and follows DeepMD's extended-region
    # convention), so only the sum identity is required of each.
    b_atom_virial = reduce_ext(b["extended_virial"], nframes, nloc, mapping)
    for name, av, ref in [("forward", a["atom_virial"], a["virial"]),
                          ("forward_lower", b_atom_virial, b["virial"].unsqueeze(1))]:
        got = av.detach().double().sum(1).reshape(nframes, -1)
        exp = ref.detach().double().reshape(nframes, -1)
        d = float((got - exp).abs().max())
        rel = d / max(float(exp.abs().max()), 1e-30)
        print(f"   sum(atom_virial) == virial [{name:13s}] max|d|={d:.3e}  rel={rel:.2e}")
        worst = max(worst, rel)
    dav = float((a["atom_virial"].detach().double() - b_atom_virial.detach().double()).abs().max())
    print(f"   per-atom split differs by {dav:.3e} (convention: raw local vs extended)")
    ok = worst < 1e-10

    # ------------------------------------------------- 2. finite-difference arbiter
    shear = SHEAR.to(dev)

    def energies(eps):
        """Total energy at this strain, from each route.

        Each route gets its own copy: forward_common flips requires_grad on the
        coordinates it is handed (in place), and a tensor that already carries a
        grad history is no longer a leaf, which forward_lower rejects.
        """
        r, bx = strained(coord, box, shear, eps)
        ext = build_extended(model, r.clone(), atype, bx.clone())
        a_out = model(r.clone(), atype, bx.clone(), do_atomic_virial=False)
        b_out = call_lower(model, *ext, bx.clone())
        return (
            float(a_out["energy"].detach().sum()),
            float(b_out["energy"].detach().sum()),
        )

    zero = torch.zeros(3, 3, dtype=torch.float64, device=dev)
    r0, b0 = strained(coord, box, shear, zero)
    with torch.enable_grad():
        ext0 = build_extended(model, r0.clone(), atype, b0.clone())
        out_b = call_lower(model, *ext0, b0.clone())
        v_b = out_b["virial"].detach().reshape(3, 3)
        v_a = (
            model(r0.clone(), atype, b0.clone(), do_atomic_virial=True)["virial"]
            .detach()
            .reshape(3, 3)
        )

    fd_a = torch.zeros(3, 3, dtype=torch.float64, device=dev)
    fd_b = torch.zeros(3, 3, dtype=torch.float64, device=dev)
    for i in range(3):
        for j in range(3):
            ep = torch.zeros(3, 3, dtype=torch.float64, device=dev)
            ep[i, j] = EPS
            em = -ep
            ea_p, eb_p = energies(ep)
            ea_m, eb_m = energies(em)
            fd_a[i, j] = -(ea_p - ea_m) / (2 * EPS)
            fd_b[i, j] = -(eb_p - eb_m) / (2 * EPS)

    # negative control: the short-range channel alone cannot reproduce this
    sr_virial = (
        model.forward_common_lower(*ext0, do_atomic_virial=True)["energy_derv_c_redu"]
        .detach()
        .squeeze(-2)
        .reshape(3, 3)
    )
    scale = max(1.0, float(fd_a.abs().max()))
    checks = [
        ("forward       ", v_a, fd_a),
        ("forward_lower ", v_b, fd_b),
        ("short-range   ", sr_virial, fd_a),
    ]
    for name, v, fd in checks:
        d = (v - fd).abs().max().item()
        print(f"   |virial - FD| [{name}] = {d:.6e}  rel={d / scale:.2e}  "
              f"antisym = {(v - v.T).abs().max().item():.3e}")
    err_a = float((v_a - fd_a).abs().max())
    err_b = float((v_b - fd_b).abs().max())
    err_sr = float((sr_virial - fd_a).abs().max())
    print(f"   LR contribution (SR vs full)          = {err_sr:.6e}")
    print(f"   FD antisym = {(fd_a - fd_a.T).abs().max().item():.3e} (must be ~0)")
    print(f"   A - B virial = {(v_a - v_b).abs().max().item():.3e}")
    ok = ok and err_a < 1e-5 * scale and err_b < 1e-5 * scale
    ok = ok and err_sr > 10 * err_a  # the control must actually separate
    ok = ok and float((v_a - v_a.T).abs().max()) < 1e-6 * scale
    ok = ok and float((v_b - v_b.T).abs().max()) < 1e-6 * scale

    # ------------------------------------------------------- 3. the cell is required
    try:
        call_lower(model, ext_coord, ext_atype, nlist, mapping, None)
        print("   !! eager: box=None did NOT raise")
        ok = False
    except Exception as e:  # noqa: BLE001 - report whatever it raised
        print(f"   eager: box=None -> {type(e).__name__}: {str(e)[:60]}...")

    # ------------------------------------------------------------ 4. scripted module
    try:
        scripted = torch.jit.script(model)
    except Exception as e:  # noqa: BLE001
        print(f"torch.jit.script raised {type(e).__name__}: {e}")
        print("RESULT: FAIL")
        return 1
    print(f"scripted: forward_lower={hasattr(scripted, 'forward_lower')} "
          f"need_lower_box={scripted.need_lower_box()}")
    print(f"   schema: {scripted.forward_lower.schema}")
    ok = ok and hasattr(scripted, "forward_lower") and scripted.need_lower_box() is True
    ok = ok and "box" in str(scripted.forward_lower.schema)
    with torch.enable_grad():
        s = call_lower(scripted, ext_coord.clone(), ext_atype.clone(), nlist.clone(),
                       mapping.clone(), box.clone())
    for key in KEYS:
        d = float((s[key].detach().double() - b[key].detach().double()).abs().max())
        print(f"   scripted vs eager {key:18s} max|d| = {d:.3e}")
        ok = ok and d < 1e-12
    try:
        call_lower(scripted, ext_coord, ext_atype, nlist, mapping, None)
        print("   !! scripted: box=None did NOT raise")
        ok = False
    except Exception as e:  # noqa: BLE001
        print(f"   scripted: box=None -> {type(e).__name__}: {str(e)[:60]}...")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
