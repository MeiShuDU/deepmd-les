# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any, Dict, Optional,
)
import torch
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
    EnergyFittingNetDirect,
    InvarFitting,
)
from deepmd.pt.model.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel import FittingOutputDef
from deepmd.dpmodel.output_def import OutputVariableDef, OutputVariableCategory


class HybridLESAtomicModel(DPAtomicModel):
    def __init__(
        self, descriptor: Any, fitting: Any, type_map: Any, les_params : Any, **kwargs: Any
    ) -> None:
        if not (
            isinstance(fitting, EnergyFittingNet)
            or isinstance(fitting, EnergyFittingNetDirect)
            or isinstance(fitting, InvarFitting)
        ):
            raise TypeError(
                "fitting must be an instance of EnergyFittingNet, EnergyFittingNetDirect or InvarFitting for DPEnergyAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)
        self.les_params = les_params

        from les import Les
        self.les_model = Les(les_arguments=les_params or {})