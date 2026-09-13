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
from deepmd.pt.model.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.pt.utils.env import (
    GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.dpmodel import FittingOutputDef
from deepmd.dpmodel.output_def import OutputVariableDef, OutputVariableCategory


@BaseAtomicModel.register("hybrid_ener")
class HybridLESAtomicModel(DPAtomicModel):
    """短程 DeePMD 原子模型 + LES 长程模块。

    与 DPAtomicModel 的唯一结构性差异是额外持有一个 LES 子模块
    (``self.les_model``)。该子模块是标准的 ``nn.Module``:
    - 其可训练权重随 ``state_dict()`` 一起保存/加载 (训练 ckpt 无需特殊处理);
    - ``serialize()`` 显式把权重与 ``les_params`` 一并写入, 使
      ``dp convert-back`` / ``BaseModel.deserialize`` 这类不经过 state_dict
      的链路也能重建出可用的长程通道。
    """

    def __init__(
        self, descriptor: Any, fitting: Any, type_map: Any,
        les_params: Optional[Dict[str, Any]] = None, **kwargs: Any
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
        self.les_params = dict(les_params) if les_params else {}

        from les import Les
        self.les_model = Les(les_arguments=self.les_params)
        # 显式对齐 DeepMD 全局浮点精度: 描述符/坐标经 input_type_cast 与
        # GLOBAL_PT_FLOAT_PRECISION 对齐 (默认 float64)。旧实现通过在
        # hybridles_model 模块顶部 torch.set_default_tensor_type(DoubleTensor)
        # 使 LES 的 nn.Linear 恰好为 double, 那是进程级全局副作用; 这里改为
        # 局部 cast, 效果等价且不污染调用方 (支持 float32 全局精度场景)。
        self.les_model.to(dtype=GLOBAL_PT_FLOAT_PRECISION)

        # 元素序数查找表: FixedCharges/AtomicAlpha 需要按原子序数取值, 早期实现
        # 用 Python dict + .item() 逐元素查表, 无法被 torch.jit.script。这里预先把
        # type_map 映射成定长张量, forward 中直接张量索引, 既 scriptable 也更快。
        # persistent=False: 该表由 type_map 完全决定, 不进 state_dict, 因此不会
        # 改变既有 ckpt 的键集合 (旧 ckpt 仍可严格加载)。
        from les.module import type2number
        self.register_buffer(
            "element_numbers",
            type2number.type2num(list(type_map)).to(torch.int64),
            persistent=False,
        )

    def serialize(self) -> dict:
        """序列化: 在 DPAtomicModel 的基础上补齐 LES 的配置与权重。

        与描述符/拟合网络一致, LES 权重以 numpy 数组形式放在 ``@variables``
        中; ``les_params`` 是构造 LES 子模块所需的全部超参 (含 dim_descrpt),
        因此序列化结果足以重建一个可用的长程通道。
        """
        dd = super().serialize()
        dd.update(
            {
                "type": "hybrid_ener",
                "les_params": self.les_params,
                "@variables": {
                    **dd.get("@variables", {}),
                    "les_model": {
                        kk: to_numpy_array(vv)
                        for kk, vv in self.les_model.state_dict().items()
                    },
                },
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data: dict) -> "HybridLESAtomicModel":
        data = data.copy()
        variables = dict(data.get("@variables") or {})
        les_variables = variables.pop("les_model", None)
        data["@variables"] = variables
        obj = super().deserialize(data)
        if les_variables is not None:
            obj.les_model.load_state_dict(
                {kk: to_torch_tensor(vv) for kk, vv in les_variables.items()}
            )
        return obj
