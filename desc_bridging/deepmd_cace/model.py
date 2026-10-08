"""Compose the DeepMD descriptor with CACE energy and optional Ewald heads."""

import torch
from deepmd.pt.model.descriptor.se_a import DescrptSeA
from cace.models.atomistic import NeuralNetworkPotential
from cace.modules.atomwise import Atomwise
from cace.modules.ewald import EwaldPotential
from cace.modules.feature_mix import FeatureAdd
from cace.modules.forces import Forces
from cace.modules.preprocess import Preprocess

from .charge_eq_latent import ChargeEqLatent
from .descriptor import DeepmdSeAInput

# the kernel options long_range.ewald may carry that ChargeEqLatent accepts: the
# rest of EwaldPotential's surface (external_field, charge_neutral_lambda) has no
# meaning for a module that never calls the kernel's own forward
LONG_RANGE_KERNEL_KEYS = (
    "dl",
    "sigma",
    "exponent",
    "remove_self_interaction",
    "feature_key",
    "output_key",
    "aggregation_mode",
)
CHARGE_EQ_KEYS = ("enabled", "regularization_weight", "total_charge", "total_charge_key")


def build_long_range_kernel(long_range):
    """The module that turns latent charges into the long-range energy.

    Plain ``EwaldPotential`` by default. With ``long_range.charge_eq_latent``
    enabled, ``ChargeEqLatent`` takes its place: same kernel, same emitted
    ``ewald_potential``, plus a constrained equalization of the charges first.
    Both are parameter-free, so a checkpoint of either loads into the other.
    """
    options = dict(long_range.get("ewald", {}))
    options.setdefault("feature_key", "q")
    options.setdefault("output_key", "ewald_potential")
    options.setdefault("aggregation_mode", "sum")
    equalization = long_range.get("charge_eq_latent", {})
    if not equalization.get("enabled", False):
        return EwaldPotential(**options)
    unsupported = sorted(set(options) - set(LONG_RANGE_KERNEL_KEYS))
    if unsupported:
        raise ValueError(
            "long_range.ewald options not supported by charge_eq_latent: "
            + ", ".join(unsupported)
        )
    # the mapping is spelled out because the two modules name their outputs
    # differently: for EwaldPotential ``output_key`` *is* the long-range energy,
    # while for ChargeEqLatent it is the equalized charges and the energy is
    # ``ewald_key``. Passing the ewald dict straight through would make the
    # charges overwrite the energy under the same key.
    return ChargeEqLatent(
        dl=options.get("dl", 2.0),
        sigma=options.get("sigma", 1.0),
        exponent=options.get("exponent", 1),
        feature_key=options["feature_key"],
        ewald_key=options["output_key"],
        aggregation_mode=options["aggregation_mode"],
        remove_self_interaction=options.get("remove_self_interaction", True),
        regularization_weight=float(equalization.get("regularization_weight", 0.1)),
        total_charge=float(equalization.get("total_charge", 0.0)),
        total_charge_key=equalization.get("total_charge_key", "system_charge"),
    )


class _MergeCombinedPotential(torch.nn.Module):
    def __init__(self, combined):
        super().__init__()
        self.combined = combined
        self.required_derivatives = list(combined.required_derivatives)
        self.model_outputs = ["CACE_energy"]

    def forward(
        self,
        data,
        training=False,
        compute_stress=False,
        compute_virials=False,
        output_index=None,
    ):
        outputs = self.combined(
            data,
            training=training,
            compute_stress=compute_stress,
            compute_virials=compute_virials,
            output_index=output_index,
        )
        data.update(outputs)
        return data


def build_model(cace_model_config, descriptor, device):
    type_map = cace_model_config["type_map"]
    fitting = cace_model_config.get("fitting_net", {})
    long_range = cace_model_config.get("long_range", {})
    combine_potentials = bool(long_range.get("combine_potentials", False))
    share_descriptor = bool(long_range.get("share_descriptor", False))
    node_features = DeepmdSeAInput(descriptor, type_map)
    input_width = descriptor.get_dim_out()

    energy_head = Atomwise(
        n_in=input_width,
        n_layers=int(fitting.get("n_layers", 3)),
        n_hidden=fitting.get("n_hidden", [32, 16]),
        feature_key="node_feats",
        output_key=(
            "CACE_energy"
            if not long_range.get("enabled", False) or combine_potentials
            else "SR_energy"
        ),
        use_batchnorm=bool(fitting.get("use_batchnorm", False)),
        add_linear_nn=bool(fitting.get("add_linear_nn", True)),
        bias=bool(fitting.get("bias", True)),
    )
    output_modules = [energy_head]

    if long_range.get("enabled", False):
        charge = long_range.get("charge_net", {})
        charge_head = Atomwise(
            n_in=input_width,
            n_out=int(charge.get("n_out", 1)),
            n_layers=int(charge.get("n_layers", 3)),
            n_hidden=charge.get("n_hidden", [24, 12]),
            feature_key="node_feats",
            output_key="tot_q",
            per_atom_output_key="q",
            residual=False,
            add_linear_nn=True,
            bias=bool(charge.get("bias", False)),
        )
        ewald = build_long_range_kernel(long_range)

        if combine_potentials:
            if share_descriptor:
                sr_model = NeuralNetworkPotential(
                    representation=None,
                    input_modules=[],
                    output_modules=[energy_head],
                )
                lr_model = NeuralNetworkPotential(
                    representation=None,
                    input_modules=[],
                    output_modules=[charge_head, ewald],
                )
                from cace.models.combined import CombinePotential

                combined = CombinePotential(
                    [sr_model, lr_model],
                    [
                        {"CACE_energy": "CACE_energy"},
                        {
                            "CACE_energy": "ewald_potential",
                            "weight": float(long_range.get("weight", 0.02)),
                        },
                    ],
                )
                combined_output = _MergeCombinedPotential(combined)
                return NeuralNetworkPotential(
                    representation=None,
                    input_modules=[Preprocess(), node_features],
                    output_modules=[
                        combined_output,
                        Forces(energy_key="CACE_energy", forces_key="CACE_forces"),
                    ],
                ).to(device)

            sr_model = NeuralNetworkPotential(
                representation=None,
                input_modules=[Preprocess(), node_features],
                output_modules=[
                    energy_head,
                    Forces(energy_key="CACE_energy", forces_key="CACE_forces"),
                ],
            )
            lr_model = NeuralNetworkPotential(
                representation=None,
                input_modules=[Preprocess(), node_features],
                output_modules=[
                    charge_head,
                    ewald,
                    Forces(energy_key="ewald_potential", forces_key="ewald_forces"),
                ],
            )
            from cace.models.combined import CombinePotential

            combined = CombinePotential(
                [sr_model, lr_model],
                [
                    {"CACE_energy": "CACE_energy", "CACE_forces": "CACE_forces"},
                    {
                        "CACE_energy": "ewald_potential",
                        "CACE_forces": "ewald_forces",
                        "weight": float(long_range.get("weight", 0.02)),
                    },
                ],
            )
            return combined.to(device)

        total_energy = FeatureAdd(
            feature_keys=["SR_energy", "ewald_potential"],
            output_key="CACE_energy",
        )
        output_modules.extend([charge_head, ewald, total_energy])

    output_modules.append(Forces(energy_key="CACE_energy", forces_key="CACE_forces"))
    return NeuralNetworkPotential(
        representation=None,
        input_modules=[Preprocess(), node_features],
        output_modules=output_modules,
    ).to(device)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != "deepmd-cace-state-dict-1":
        raise ValueError(f"unsupported DeepMD-CACE checkpoint: {path}")
    config = checkpoint["config"]
    descriptor = DescrptSeA(**{
        key: value for key, value in config["model"]["descriptor"].items()
        if key != "type"
    })
    model = build_model(config["model"], descriptor, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model