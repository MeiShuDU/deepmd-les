"""CPU regression checks and a generated mixed CONP/CONQ training fixture.

Run with CUDA_VISIBLE_DEVICES="" PYTHONPATH=desc_bridging python
desc_bridging/check_charge_eq_fixed_groups.py --output-dir /tmp/charge-eq-smoke.
The generated labels only exercise the training pipeline, not physical accuracy.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from deepmd_cace.charge_eq_latent import ChargeEqLatent, equalize_charges
from deepmd_cace.ewald_matrix import EwaldMatrixPotential, coulomb_matrix
from deepmd_cace.config import load_config
from deepmd_cace.data import load_split
from deepmd_cace.model import build_model, load_checkpoint
from deepmd_cace.train import build_descriptor, compute_descriptor_stats, run


def expect_error(function, message):
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def generate_fixture(root):
    rng = np.random.default_rng(14)
    base = np.array([[1., 1., 1.], [2., 1., 1.], [1., 1., 5.],
                     [2., 1., 5.], [1., 2., 3.], [2., 2., 3.]])
    specs = []
    for mode in ("CONP", "CONQ"):
        for split, nframes in (("train", 2), ("valid", 1)):
            folder = root / mode / split
            arrays = folder / "set.000"
            arrays.mkdir(parents=True, exist_ok=True)
            (folder / "type.raw").write_text("0 0 0 0 0 0\n")
            (folder / "type_map.raw").write_text("H\n")
            displacement = rng.normal(0, .025, (nframes, 6, 3))
            np.save(arrays / "coord.npy", (base + displacement).reshape(nframes, -1))
            np.save(arrays / "box.npy", np.tile(np.diag([8., 8., 12.]).reshape(1, 9), (nframes, 1)))
            np.save(arrays / "energy.npy", .5 * (displacement ** 2).sum(axis=(1, 2)))
            np.save(arrays / "force.npy", -displacement.reshape(nframes, -1))
        group_path = root / mode / "groups.json"
        group_path.write_text(json.dumps({"electrodeA": [1, 2], "electrodeB": [3, 4]}))
        electrodes = ([dict(type="CONP", group1="electrodeA", group2="electrodeB", delta_V=.02)]
                      if mode == "CONP" else [dict(type="CONQ", group=g, total_charge=v)
                                             for g, v in (("electrodeA", .1), ("electrodeB", -.05))])
        specs.append(dict(system=f"{mode}/train", system_aliases=[f"{mode}/valid"],
                          group_definition=f"{mode}/groups.json", electrode_mode=mode,
                          electrode_constraints=electrodes,
                          global_constraints=[dict(type="CONQ", group="all", total_charge=0.)]))
    config = {
        "model": {"type_map": ["H"], "arm": "lr",
                  "descriptor": {"type": "se_a", "rcut": 4., "rcut_smth": .5,
                                 "sel": [8], "neuron": [4, 8], "axis_neuron": 2,
                                 "seed": 3, "precision": "float64"},
                  "fitting_net": {"n_layers": 3, "n_hidden": [8, 4]},
                  "long_range": {"charge_net": {"n_layers": 3, "n_hidden": [8, 4]},
                                 "ewald": {"dl": 3., "sigma": 1., "remove_self_interaction": False},
                                 "charge_eq_latent": {"enabled": True, "regularization_weight": .1,
                                                      "learnable_kappa": True, "systems": specs}}},
        "loss": {"energy_weight": 1., "force_weight": 1.},
        "training": {"training_data": {"systems": ["CONP/train", "CONQ/train"], "batch_size": 4},
                     "validation_data": {"systems": ["CONP/valid", "CONQ/valid"], "batch_size": 2},
                     "epochs": 1, "device": "cpu", "seed": 3, "warmup_steps": 0,
                     "optimizer": {"lr": .001}, "save_freq": 1, "disp_freq": 1,
                     "output_dir": "run"}}
    path = root / "input.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def solver_checks():
    torch.manual_seed(7)
    dtype = torch.float64
    a = torch.eye(4, dtype=dtype)
    proposal = torch.randn(4, 2, dtype=dtype)
    q, dual = equalize_charges(a, proposal, .1, .3)
    torch.testing.assert_close(q.sum(), q.new_tensor(.3))
    # Roundoff antisymmetry must not affect the quadratic minimizer.
    skew = torch.zeros_like(a)
    skew[0, 1], skew[1, 0] = 1e-5, -1e-5
    symmetric_q, _ = equalize_charges(a + skew, proposal, .1, .3)
    torch.testing.assert_close(symmetric_q, q)
    torch.testing.assert_close(1.2 * q - .2 * proposal + dual, torch.zeros_like(q))
    c = torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]], dtype=dtype)
    q, _ = equalize_charges(a, proposal[:, :1], 0., constraint_matrix=c,
                            constraint_values=[.2, -.2])
    torch.testing.assert_close(c @ q[:, 0], q.new_tensor([.2, -.2]))
    # A singular unconstrained Hessian can still define a unique constrained minimum.
    q, _ = equalize_charges(torch.diag(torch.tensor([0., 1.], dtype=dtype)),
                            torch.ones(2, 1, dtype=dtype), 0., constraint_matrix=torch.tensor([[1., 0.]], dtype=dtype),
                            constraint_values=[.2])
    torch.testing.assert_close(q[:, 0], q.new_tensor([.2, 0.]))
    expect_error(lambda: equalize_charges(-a, proposal, .1), "not strictly convex")
    expect_error(lambda: equalize_charges(a, proposal, .1, constraint_matrix=c.repeat(2, 1),
                                          constraint_values=[0.] * 4), "linearly dependent")
    def differentiable(x):
        matrix = a + torch.outer(x, x)
        rows = torch.stack([torch.ones_like(x), x @ matrix])
        return equalize_charges(matrix, x[:, None], .1, constraint_matrix=rows,
                                constraint_values=[0., .1], kappa_inv=torch.nn.functional.softplus(x))[0]
    x = torch.tensor([.1, -.2, .4, -.3], dtype=dtype, requires_grad=True)
    assert torch.autograd.gradcheck(differentiable, (x,))
    assert torch.autograd.gradgradcheck(differentiable, (x,))
    print("solver: legacy channels, metallic limit, invalid constraints, first/second derivatives OK")


def kernel_api_checks():
    from cace.modules.ewald import EwaldPotential

    kernel = EwaldMatrixPotential()
    r = torch.zeros(2, 3)
    cell = torch.eye(3)
    expected = torch.eye(2)
    with patch.object(EwaldPotential, "compute_coulomb_matrix", create=True,
                      return_value=expected) as native_matrix:
        assert coulomb_matrix(kernel, r, cell) is expected
        native_matrix.assert_called_once_with(r, cell)
    print("Ewald adapter: native matrix API delegation OK")


def module_checks(config):
    specs = config["model"]["long_range"]["charge_eq_latent"]["systems"]
    module = ChargeEqLatent(systems=specs, learnable_kappa=True, kappa_in=3,
                            remove_self_interaction=False, dl=3.).double()
    r = torch.tensor([[1., 1., 1.], [2., 1., 1.], [1., 1., 5.],
                      [2., 1., 5.], [1., 2., 3.], [2., 2., 3.]], dtype=torch.float64)
    cell = torch.diag(r.new_tensor([8., 8., 12.]))
    # Each frame obtains one operator from Ewald and reuses it throughout forward.
    with patch.object(module.ep, "compute_coulomb_matrix",
                      wraps=module.ep.compute_coulomb_matrix) as matrix_call:
        module(dict(positions=r, cell=cell, q=.1 * torch.sin(r[:, :1]),
                    node_feats=torch.sin(r), charge_eq_system=torch.tensor([0])))
        assert matrix_call.call_count == 1
    def evaluate(positions, system_id):
        raw = .1 * torch.sin(positions[:, :1])
        data = dict(positions=positions, cell=cell, q=raw,
                    node_feats=torch.sin(positions), charge_eq_system=torch.tensor([system_id]))
        result = module(data)
        assert result["q"] is raw
        a = module._coulomb_matrix(positions, cell)
        c, values = module._frame_constraints(result, 0, torch.ones(6, dtype=torch.bool), raw, a, 0.)
        torch.testing.assert_close(c @ result["q_eq"][:, 0], values, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(result["ewald_potential"].sum(), .5 * (result["q_eq"] * (a @ result["q_eq"])).sum())
        return result["ewald_potential"].sum()
    for system_id in (0, 1):
        positions = r.clone().requires_grad_()
        energy = evaluate(positions, system_id)
        forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
        finite = torch.zeros_like(r)
        step = 1e-5
        for atom in range(6):
            for axis in range(3):
                shift = torch.zeros_like(r)
                shift[atom, axis] = step
                finite[atom, axis] = -(evaluate(r + shift, system_id) - evaluate(r - shift, system_id)) / (2 * step)
        torch.testing.assert_close(forces, finite, rtol=2e-5, atol=1e-8)
        (energy.square() + forces.square().sum()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.kappa_head.parameters())
        assert module.kappa_head[-1].weight.grad.abs().max() > 0
        module.zero_grad()
    module.slab_axis = 2
    for system_id in (0, 1):
        positions = r.clone().requires_grad_()
        energy = evaluate(positions, system_id)
        forces = -torch.autograd.grad(energy, positions)[0]
        finite = torch.zeros_like(r)
        for atom in range(6):
            for axis in range(3):
                shift = torch.zeros_like(r)
                shift[atom, axis] = step
                finite[atom, axis] = -(evaluate(r + shift, system_id) - evaluate(r - shift, system_id)) / (2 * step)
        torch.testing.assert_close(forces, finite, rtol=2e-5, atol=1e-8)
    nonneutral = copy.deepcopy(specs)
    nonneutral[0]["constraints"][-1]["total_charge"] = .1
    nonneutral[0]["constraints"][-1]["indices"] = list(range(6))
    bad_module = ChargeEqLatent(systems=nonneutral, slab_axis=2, remove_self_interaction=False)
    expect_error(lambda: bad_module(dict(positions=r, cell=cell, q=r[:, :1],
                                         charge_eq_system=torch.tensor([0]))), "neutrality")
    expect_error(lambda: module(dict(positions=r, cell=cell, q=r[:, :1], node_feats=r)), "metadata")
    print("module: both modes, neutrality, Coulomb/slab forces and hardness gradients OK")


def integration_checks(path):
    config = load_config(path)
    specs = config["model"]["long_range"]["charge_eq_latent"]["systems"]
    loader, stats = load_split(config["training"]["training_data"]["systems"], ["H"], 4., 4,
                               False, charge_eq_systems=specs)
    assert next(iter(loader))["charge_eq_system"].tolist() == [0, 0, 1, 1]
    descriptor = build_descriptor(config["model"]["descriptor"])
    compute_descriptor_stats(descriptor, stats)
    disabled = copy.deepcopy(config["model"])
    disabled["long_range"]["charge_eq_latent"]["enabled"] = False
    disabled["long_range"]["charge_net"]["n_out"] = 2
    plain = build_model(disabled, descriptor, "cpu")
    assert not any(isinstance(module, ChargeEqLatent) for module in plain.modules())
    for combined, shared in ((False, False), (False, True), (True, False), (True, True)):
        model_config = copy.deepcopy(config["model"])
        model_config["long_range"].update(combine_potentials=combined, share_descriptor=shared)
        model = build_model(model_config, descriptor, "cpu")
        out = model(next(iter(loader)).to_dict(), training=True)
        assert torch.isfinite(out["CACE_forces"]).all()
        (out["CACE_energy"].square().mean() + out["CACE_forces"].square().mean()).backward()
        filename = path.parent / f"layout-{int(combined)}-{int(shared)}.pt"
        torch.save(dict(format="deepmd-cace-state-dict-1", config=dict(config, model=model_config),
                        state_dict=model.state_dict()), filename)
        reloaded = load_checkpoint(filename)
        prediction = reloaded(next(iter(loader)).to_dict())
        torch.testing.assert_close(prediction["CACE_energy"], out["CACE_energy"])
        torch.testing.assert_close(prediction["CACE_forces"], out["CACE_forces"])
    print("integration: mixed-system batch, all four layouts, force-loss backward and portable checkpoint reload OK")
    return config


def schema_checks(path):
    original = json.loads(path.read_text())
    invalid = path.parent / "invalid.json"
    def check(edit, message):
        config = copy.deepcopy(original)
        edit(config["model"]["long_range"]["charge_eq_latent"])
        invalid.write_text(json.dumps(config))
        expect_error(lambda: load_config(invalid), message)
    check(lambda eq: eq["systems"][0].update(electrode_mode="CONQ"), "electrode constraints must use")
    check(lambda eq: eq["systems"][0].update(system_aliases=[]), "missing charge_eq_latent")
    check(lambda eq: eq["systems"][1]["global_constraints"].append(
        dict(type="CONQ", group="all", total_charge=1.)), "redundant or contradictory")
    check(lambda eq: eq["systems"][0]["global_constraints"].append(
        dict(type="CONQ", group="electrodeA", total_charge=0.)), "outside electrode atoms")
    bad_groups = path.parent / "bad-groups.json"
    for indices in ([0, 1], [1, 1], [True], [7], []):
        bad_groups.write_text(json.dumps(dict(electrodeA=indices, electrodeB=[3, 4])))
        check(lambda eq: eq["systems"][0].update(group_definition="bad-groups.json"), "one-based indices")
    print("schema: electrode mode conflicts, missing systems, invalid groups, duplicate constraints OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir or Path(tempfile.mkdtemp(prefix="charge-eq-smoke-"))
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    path = generate_fixture(root)
    schema_checks(path)
    solver_checks()
    kernel_api_checks()
    config = integration_checks(path)
    module_checks(config)
    run(str(path), check_only=True)
    run(str(path), smoke=True)
    print(f"all fixed-group charge_eq_latent checks passed; fixture and training outputs: {root}")


if __name__ == "__main__":
    main()
