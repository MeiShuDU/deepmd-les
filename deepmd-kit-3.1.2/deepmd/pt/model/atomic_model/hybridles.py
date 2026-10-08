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


def _jsonable(value: Any) -> Any:
    """把 les_params 里的张量 / ndarray 归一化成可 json 序列化的形式。

    ``model.model_def_script = json.dumps(model_params)`` (见 hybridles_model.get_model)
    会序列化整份配置。``initial_guess`` / ``freeze_charge`` 既可以写成 JSON 列表
    (input.json 的写法), 也可以由程序直接传 torch.Tensor; 后者必须在此归一化,
    否则 json.dumps 直接报错。Les 内部再把它们转回张量 (见 Les._type_table)。
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().reshape(-1).tolist()
    if hasattr(value, "tolist"):  # numpy 数组 / 标量
        return value.tolist()
    return value


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
        self.les_params = _jsonable(dict(les_params) if les_params else {})

        # LES 的逐类型电荷层 (freeze_charge / initial_guess) 以 type_map 为索引基准,
        # 需要类型数与逐类型原子序数 (氧化数基线按真实原子序数查表)。这里在构造
        # Les 之前注入, 使 Les 自身只需持有 int 与张量, 不必持有 type_map 字符串
        # 列表 (TorchScript 更友好)。
        from les.module import type2number
        element_numbers = type2number.type2num(list(type_map))
        self.les_params.setdefault("ntypes", len(type_map))
        self.les_params.setdefault("element_numbers", [int(z) for z in element_numbers])

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
        self.register_buffer(
            "element_numbers",
            element_numbers.to(torch.int64),
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
