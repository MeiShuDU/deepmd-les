# SPDX-License-Identifier: LGPL-3.0-or-later
"""HybridLESModel: 短程 DeePMD 网络 + LES 长程静电修正.

设计约定与已知限制 (供审查追溯):
1. 总能量 E = E_SR (rcut 内截断) + E_LR (LES/Ewald)。loss 训练时 energy/force
   通过同一张计算图反传, 二者自洽。
2. 预测 dict 中 ``virial`` / ``atom_virial`` 同时包含短程与长程两部分。长程项 =
   原子项 (长程力 ⊗ 坐标) + 显式晶胞项 (-(dE_LR/dcell) cell^T, 覆盖 Ewald 对
   volume/倒格矢的显式依赖); 二者与 DeepMD 短程约定 V = -dE/deps 一致, 因此可直接
   相加, pref_v>0 / NPT 下的能量-应力关系自洽。
3. 本模型前向可被 torch.jit.script: 帧循环已向量化为单次带 batch 的 LES 调用,
   autograd.grad 使用 List 入参形式, 记录/调试分支由 torch.jit.is_scripting() 隔离。
   ``dp freeze`` / jit 推理链路可用 (eager 路径行为不变)。
4. 不设置模块级 torch.set_default_tensor_type (全局副作用), 数值精度由
   deepmd GLOBAL_PT_FLOAT_PRECISION 统一控制。
"""
from typing import Any, Optional, Dict, List
import copy
import json
import logging

import torch

from deepmd.pt.model.model.model import BaseModel
from deepmd.pt.model.model.make_model import make_model
from deepmd.pt.model.model.dp_model import DPModelCommon
from deepmd.pt.utils.env import GLOBAL_PT_FLOAT_PRECISION
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list
from les import Les
from les.module import type2number
from deepmd.pt.model.atomic_model.hybridles import HybridLESAtomicModel
from deepmd.pt.model.descriptor import BaseDescriptor
from deepmd.pt.model.task.fitting import BaseFitting

log = logging.getLogger(__name__)

HybridLESModel_ = make_model(HybridLESAtomicModel)

@BaseModel.register("hybrid_ener")
class HybridLESModel(DPModelCommon, HybridLESModel_):
    model_type = "hybrid_ener"

    def __init__(
        self,
        descriptor=None,
        fitting=None,
        type_map: Optional[list[str]] = None,
        les_params: Optional[Dict[str, Any]] = None,
        atomic_model_: Optional[HybridLESAtomicModel] = None,
        **kwargs,
    ) -> None:
        # 两条构造路径, 与 make_model 生成的 CM 保持一致:
        # (1) get_model(): 传入 descriptor/fitting/type_map/les_params;
        # (2) deserialize(): 传入已反序列化好的 atomic_model_。
        # 二者统一在下方收敛为「持有一个 HybridLESAtomicModel」。
        if atomic_model_ is None:
            atomic_model_ = HybridLESAtomicModel(
                descriptor=descriptor,
                fitting=fitting,
                type_map=type_map,
                les_params=les_params,
            )
        HybridLESModel_.__init__(self, atomic_model_=atomic_model_, **kwargs)
        DPModelCommon.__init__(self)
        # 内部精度: 直接持有 dtype 实例属性 (DeepMD descriptor 用 self.prec 同款写法),
        # 避免 forward 中引用全局 GLOBAL_PT_FLOAT_PRECISION (torch.jit.script 无法
        # 把 dtype 全局值作为闭包常量)。
        self.pt_prec = GLOBAL_PT_FLOAT_PRECISION
        # NOTE: descriptor/type_map 统一通过 atomic_model 访问, 避免同一批参数
        # 被注册两次 (state_dict 中会出现 descriptor.* 与 atomic_model.descriptor.*
        # 双份键, 造成 ckpt 体积翻倍且易引起追溯混乱)。
        les_params = atomic_model_.les_params or {}
        type_map = atomic_model_.type_map
        self.les_log_freq = les_params.get("log_freq", 100)
        self.les_verbose = bool(les_params.get("verbose", False))
        self._les_step_counter = 0
        # 元素名校验: type2num 对未知名称返回 404 占位 (原代码为静默行为),
        # 固定电荷/原子极化率依赖真实的原子序数, 早期显式报错以便追溯。
        if (
            les_params.get("use_fixed_atomic_charges", False)
            or les_params.get("use_atomic_alpha", False)
        ):
            am = type2number.type2num(list(type_map))
            if (am == 404).any():
                unknown = [
                    t
                    for t, z in zip(type_map, am.tolist())
                    if z == 404
                ]
                raise ValueError(
                    f"type_map contains non-element names {unknown}, which cannot be "
                    "mapped to atomic numbers for FixedCharges/AtomicAlpha. "
                    "Use standard element symbols (e.g. O, H)."
                )
        self._register_les_grad_hooks()

    def _register_les_grad_hooks(self) -> None:
        """给 LES 参数注册 backward hook 以追溯梯度 (仅在 backward 后触发)."""
        if not self.les_verbose:
            return

        def make_hook(name: str):
            counter = {"n": 0}
            def hook(grad: torch.Tensor) -> None:
                counter["n"] += 1
                # 节流: 每 log_freq 次 backward 打印一次 (首次与之后每 freq 次)
                if (counter["n"] - 1) % self.les_log_freq != 0:
                    return
                if grad is None:
                    log.info(f"[LES-grad] {name}: grad=None")
                else:
                    g = grad.detach().float()
                    log.info(
                        f"[LES-grad] {name}: |grad|={g.norm().item():.4e} "
                        f"mean={g.mean().item():.4e}"
                    )
            return hook

        for name, p in self.atomic_model.les_model.named_parameters():
            if p.requires_grad:
                p.register_hook(make_hook(name))

    @torch.jit.unused
    def _maybe_log_les_decomposition(
        self,
        model_ret: Dict[str, torch.Tensor],
        e_lr_total: torch.Tensor,
        force_sr: Optional[torch.Tensor],
        force_lr_total: Optional[torch.Tensor],
    ) -> None:
        """周期性输出 SR/LR 能量与力的分解统计, 便于追溯 LES 是否在学习.

        仅在 les_params.verbose=True 且达到 log_freq 步时打印, 频率与 LES 内部
        日志 (les.log) 保持一致。
        """
        if not self.les_verbose:
            return
        self._les_step_counter += 1
        if self._les_step_counter % self.les_log_freq != 0 and self._les_step_counter != 1:
            return
        e_sr = model_ret["energy_redu"]  # [nframes]
        nloc = e_lr_total.shape[1] if e_lr_total.dim() == 2 else None
        e_lr_1d = e_lr_total
        log.info(
            "[HybridLES] step=%d | E_SR(frm0)=%.6f E_LR(frm0)=%.6f | "
            "mean|E_LR/E_SR|(batch)=%.4f | E_LR range=[%.4e, %.4e]",
            self._les_step_counter,
            float(e_sr[0].detach()),
            float(e_lr_1d[0].detach()),
            float((e_lr_1d.abs() / (e_sr.abs() + 1e-8)).mean().detach()),
            float(e_lr_1d.min().detach()),
            float(e_lr_1d.max().detach()),
        )
        if force_sr is not None and force_lr_total is not None:
            log.info(
                "[HybridLES] step=%d | RMS F_SR=%.4e RMS F_LR=%.4e "
                "|F_LR|/|F_SR|(mean)=%.4f",
                self._les_step_counter,
                float(force_sr.detach().square().mean().sqrt()),
                float(force_lr_total.detach().square().mean().sqrt()),
                float(
                    (force_lr_total.detach().norm(dim=-1).mean())
                    / (force_sr.detach().norm(dim=-1).mean() + 1e-8)
                ),
            )

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        # 1. 调用父类 forward_common 获得短程结果（能量、力等）
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )

        # 2. 重新计算描述符（用于 LES）。与短程路径保持一致: 邻居表是否按类型
        #    分组由 self.mixed_types() 决定 (之前硬编码 mixed_types=False, 对
        #    se_a 恰好成立, 对混合型描述符则会静默算错)。
        # LES 通道数值精度与短程通道统一到 GLOBAL_PT_FLOAT_PRECISION (默认
        # float64): descriptor 输出 desc 为该精度, Les 参数亦已在 atomic_model
        # 构建时 cast 到该精度, 此处统一坐标/晶胞避免 float32 输入的 dtype 冲突。
        coord = (
            coord.to(self.pt_prec)
            if coord.dtype != self.pt_prec
            else coord
        )
        if box is not None and box.dtype != self.pt_prec:
            box = box.to(self.pt_prec)

        # requires_grad 必须在重算描述符之前开启: extend_input_and_build_neighbor_list
        # 由 coord 构造 extended_coord, 若此刻 coord 不含梯度, 则 extended_coord/desc
        # 会被当作常量, 长程力会静默丢失电荷响应项 ∂E_LR/∂q·∂q/∂desc·∂desc/∂r, 只剩
        # -∂E_LR/∂r|_{q 固定}, 导致能量-力不自洽。
        if not coord.is_leaf:
            # coord 已是某个中间结果 (非叶张量), 原地置 requires_grad 会抛错;
            # 说明该模型被嵌入到需要二阶坐标梯度的外部链路, 当前实现不支持。
            # (TorchScript 无 grad_fn, 用 is_leaf 判断, 二者等价于「是否为叶张量」。)
            raise RuntimeError(
                "HybridLESModel requires leaf-like coord input to enable "
                "requires_grad_ for the long-range force autograd."
            )
        need_force = self.do_grad_r("energy")
        need_virial = self.do_grad_c("energy")
        # virial 的原子项需要长程力, 因此需要 virial 时也必须打开坐标梯度。
        need_coord_grad = need_force or need_virial
        if need_coord_grad:
            coord.requires_grad_(True)   # 确保描述符与坐标都进入长程力计算图
        else:
            coord.requires_grad_(False)
        # 显式晶胞项需要 dE_LR/dcell: Ewald 能量显式依赖 cell (volume、倒格矢、
        # Nk 网格), 这部分无法由「力 ⊗ 坐标」的原子项覆盖。
        if need_virial and box is not None:
            box.requires_grad_(True)

        rcut = self.get_rcut()
        sel = self.get_sel()
        descriptor = self.atomic_model.descriptor
        extended_coord, extended_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            coord, atype, rcut, sel, box=box, mixed_types=self.mixed_types()
        )
        desc = descriptor(extended_coord, extended_atype, nlist)[0]  # [nframes, nloc, dim]

        # 3. 计算 LES 长程能量和力
        nframes = coord.shape[0]
        nloc = coord.shape[1]

        # 帧循环已向量化: 单次调用 LES 覆盖所有帧。Ewald.forward 按 unique(batch)
        # 逐图计算并 torch.cat, 顺序与 batch=arange(repeat_interleave) 一致, 因此
        # E_lr_total 第 i 个元素仍对应第 i 帧, 与旧版逐帧循环逐位一致。坐标/描述符
        # 按 (frame, atom) 平铺, 与 extend_input_and_build_neighbor_list 的顺序一致。
        # 原子序数由 type_map 查找表 (element_numbers) 派生, 替代旧版逐帧 Python
        # list (type_map[at]), 结果相同且可被 torch.jit.script。
        batch = (
            torch.arange(nframes, dtype=torch.int64, device=coord.device)
            .repeat_interleave(nloc)
        )
        positions = coord.reshape(-1, 3)
        desc_flat = desc.reshape(-1, desc.shape[-1])
        cell_all = box.reshape(nframes, 3, 3) if box is not None else None
        atomic_numbers = self.atomic_model.element_numbers[
            atype.reshape(-1)
        ].to(device=coord.device)
        les_out = self.atomic_model.les_model(
            positions=positions,
            cell=cell_all,
            desc=desc_flat,
            batch=batch,
            compute_energy=True,
            atomic_numbers=atomic_numbers,
        )
        E_lr_total = les_out["E_lr"]  # [nframes]
        # Dict[str, Optional[Tensor]] 返回值精化为 Tensor (TorchScript 需要确定类型)
        assert E_lr_total is not None

        # 长程力必须对完整的 coord 求梯度, 而不是对切片 coord[i]: 描述符分支经
        # extend_input_and_build_neighbor_list 挂在父张量 coord 上, 若只对 coord_i
        # 求梯度, 该分支不可见, 电荷响应项 ∂E_LR/∂q·∂q/∂desc·∂desc/∂r 会被静默丢弃,
        # 只剩 -∂E_LR/∂r|_{q 固定} (已用有限差分验证: 对 coord_i 求梯度会漏掉该量)。
        # create_graph=True: 力对 LES/描述符参数可二阶反传; retain_graph=True: 为后续
        # loss.backward() 保留计算图。
        # grad_outputs 显式写成 List[Optional[Tensor]]: torch.autograd.grad 的
        # TorchScript schema 要求 List[Tensor] 入参/出参, 裸张量形式只在 eager 下可用
        # (dp freeze 需要 script, 见 hybridles_model.py 顶部约定 3)。
        grad_ones = torch.jit.annotate(
            List[Optional[torch.Tensor]], [torch.ones_like(E_lr_total)]
        )
        force_lr_total: Optional[torch.Tensor] = None
        if need_coord_grad:
            dE_lr_dcoord = torch.autograd.grad(
                [E_lr_total],
                [coord],
                grad_outputs=grad_ones,
                create_graph=True,
                retain_graph=True,
            )[0]  # [nframes, nloc, 3]
            # List[Optional[Tensor]] 取下标后精化为 Tensor
            assert dE_lr_dcoord is not None
            force_lr_total = -dE_lr_dcoord

        # 长程 virial, 与短程沿用同一约定 V = -dE/deps (eps 为固定分数坐标下的应变
        # r -> r(I+eps), h -> h(I+eps))。链式展开后有两项, 缺一不可:
        #   原子项   -sum_i r_i (x) dE_LR/dr_i   (坐标随应变缩放的部分)
        #   显式晶胞项 -(dE_LR/dh) h^T           (Ewald 独有的 volume/倒格矢依赖)
        # 短程通道只依赖坐标, 故其 virial 只有原子项 (由 DeepMD 在 extended 区域给出)。
        #
        # 指标配对必须与 DeepMD 一致 (见 model/transform_output.py task_deriv_one):
        # 短程用 einsum("...ik,...ij->...ikj", force, coord), 即 V_kj = sum_i F_ik r_ij。
        # 对应到应变导数 (dr_ij/deps_ab = r_ia delta_jb, dh_lj/deps_ab = h_la delta_jb):
        #   -dE/deps_ab = sum_i F_ib r_ia - sum_l A_lb h_la   (A = dE_LR/dh)
        # 即 einsum("...ia,...ib->...ab", coord, force) (注意与短程的写法互为转置),
        # 晶胞项同理写成 einsum("...la,...lb->...ab", box, grad_cell)。
        # 该量恒为对称张量: eps 的反对称部分对应整体刚性转动, 能量不变, 其应变导数
        # 必为零。用「力在前、坐标在后」的写法会引入一个虚假的反对称分量 (已用非对称
        # 应变的有限差分在非立方晶胞上定量验证: 误差 7.7e-7, 而错误写法误差 35.6)。
        virial_lr: Optional[torch.Tensor] = None
        atom_virial_lr: Optional[torch.Tensor] = None
        if need_virial:
            assert force_lr_total is not None
            v_cell = torch.zeros(
                nframes, 3, 3, dtype=coord.dtype, device=coord.device
            )
            if box is not None:
                grad_cell = torch.autograd.grad(
                    [E_lr_total],
                    [box],
                    grad_outputs=grad_ones,
                    create_graph=True,
                    retain_graph=True,
                    # 非周期 (box 全零) 时 Ewald 走实空间直接求和, cell 不进入计算图。
                    allow_unused=True,
                )[0]
                if grad_cell is not None:
                    v_cell = -torch.einsum(
                        "nla,nlb->nab",
                        box.reshape(nframes, 3, 3),
                        grad_cell.reshape(nframes, 3, 3),
                    )
            virial_lr = (
                torch.einsum("nia,nib->nab", coord, force_lr_total) + v_cell
            ).reshape(nframes, 9)
            # 原子 virial 只保证「逐原子求和 = 总 virial」这一恒等式: 原子项本身
            # 就是逐原子的, 而显式晶胞项是全同的, 平均摊到每个原子上。
            atom_virial_lr = (
                torch.einsum("nia,nib->niab", coord, force_lr_total)
                + (v_cell / nloc).unsqueeze(1)
            ).reshape(nframes, nloc, 1, 9)

        # 4. 合并结果
        model_predict = {}
        model_predict["energy"] = model_ret["energy_redu"] + E_lr_total
        # 短程原子能量 (不含长程): 能量损失默认用总能量 energy, 原子能仅用于
        # 追溯/原子能损失, 原子级 LES 贡献未定义, 这里保持短程值并显式说明。
        model_predict["atom_energy"] = model_ret["energy"]
        if need_force:
            # need_force => need_coord_grad, 长程力必然已计算过
            assert force_lr_total is not None
            force_sr = model_ret["energy_derv_r"].squeeze(-2)  # [nframes, nloc, 3]
            model_predict["force"] = force_sr + force_lr_total
        else:
            force_sr = None
        if need_virial:
            # need_virial 与上方长程 virial 计算同源, TS 流分析无法跨 bool flag
            # 关联 Optional, 故在此显式精化。
            assert virial_lr is not None
            # 短程 virial (DeepMD 在 extended 区域给出) + 长程 virial (原子项 + 显式
            # 晶胞项)。两者用同一约定, 因此可以直接相加。
            model_predict["virial"] = (
                model_ret["energy_derv_c_redu"].squeeze(-2) + virial_lr
            )
            if do_atomic_virial:
                assert atom_virial_lr is not None
                model_predict["atom_virial"] = (
                    model_ret["energy_derv_c"].squeeze(-3) + atom_virial_lr
                )
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]

        # 日志/调试分支 (含 f-string 格式与逐帧统计) 仅 eager 下生效; 方法本体标记为
        # @torch.jit.unused, 不进入冻结模型的计算图。
        if torch.jit.is_scripting():
            pass
        else:
            self._maybe_log_les_decomposition(
                model_ret, E_lr_total, force_sr, force_lr_total
            )
        return model_predict

    @classmethod
    def get_model(cls, model_params: dict) -> "HybridLESModel":
        """Construct a HybridLESModel from a parameter dictionary."""
        # 深拷贝以避免向 model_params 注入 ntypes/type_map/dim_descrpt 等派生键,
        # 污染调用方配置 (影响后续追溯与重复构建)。
        model_params = copy.deepcopy(model_params)
        type_map = model_params["type_map"]
        ntypes = len(type_map)

        # 1. 构建描述符
        descr_params = model_params.get("descriptor", {})
        descr_params["ntypes"] = ntypes
        descr_params["type_map"] = type_map
        descriptor = BaseDescriptor(**descr_params)

        # 2. 构建拟合网络
        fit_params = model_params.get("fitting_net", {})
        fit_params["type"] = fit_params.get("type", "ener")
        fit_params["ntypes"] = ntypes
        fit_params["type_map"] = type_map
        fit_params["mixed_types"] = descriptor.mixed_types()
        fit_params["dim_descrpt"] = descriptor.get_dim_out()
        # 如果拟合类型需要嵌入宽度，则设置
        if fit_params["type"] in ["dipole", "polar"]:
            fit_params["embedding_width"] = descriptor.get_dim_emb()
        fitting = BaseFitting(**fit_params)

        # 3. 获取 les_params
        les_params = model_params.get("les_params", {}).copy()
        les_params.setdefault("dim_descrpt", descriptor.get_dim_out())

        # 4. 实例化模型
        model = cls(
            descriptor=descriptor,
            fitting=fitting,
            type_map=type_map,
            les_params=les_params,
        )
        # 记录模型定义脚本: 供 .pt/.pth 冻结、dp show、转换等链路的元数据追溯。
        # 与其他模型分支 (见 deepmd.pt.model.model.__init__.get_standard_model)
        # 保持一致, 存为 json 字符串。
        model.model_def_script = json.dumps(model_params)
        return model
