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
from les import Les
from les.module import type2number
from deepmd.pt.model.atomic_model.hybridles import HybridLESAtomicModel, _jsonable
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
        # 与 les.py 保持一致: log_freq 非正数按「关闭日志」处理。
        # 否则下面两处 `% self.les_log_freq` (grad hook 与分解日志) 会除零。
        if self.les_log_freq <= 0:
            self.les_verbose = False
        # 长程能量整体缩放因子: 论文 cace-LES 基准以 0.01 融合短程/长程能量,
        # deepmd 原生实现以 1.0 直接相加; 该权重使 E_LR = lr_weight * E_lr_NN,
        # 并自动作用于长程力/长程 virial (它们都从缩放后的 E_lr 求导得到)。
        self.lr_weight = float(les_params.get("lr_weight", 1.0))
        self._les_step_counter = 0
        # 元素名校验: type2num 对未知名称返回 404 占位 (原代码为静默行为),
        # 固定电荷/原子极化率依赖真实的原子序数, 早期显式报错以便追溯。
        # use_fixed_charges 是 use_fixed_atomic_charges 的新名称, 二者等价触发校验。
        if (
            les_params.get("use_fixed_atomic_charges", False)
            or les_params.get("use_fixed_charges", False)
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
        # 1. 精度统一。LES 通道与短程通道共用同一个 coord/box (见第 2 步的描述符复用),
        #    因此在这里一次性统一到 GLOBAL_PT_FLOAT_PRECISION (默认 float64):
        #    descriptor 输出为该精度, Les 参数亦已在 atomic_model 构建时 cast 到该精度,
        #    统一坐标/晶胞可避免 float32 输入的 dtype 冲突。对 float64 调用方 (dp 训练/
        #    推理的常规路径) 此处的 to() 是 no-op; 对 float32 调用方, 短程通道的输入
        #    精度随之提升 (父类内部本就要 cast 到该精度, 只是不再按调用方精度回写中间量)。
        coord = (
            coord.to(self.pt_prec)
            if coord.dtype != self.pt_prec
            else coord
        )
        if box is not None and (
            box.dtype != self.pt_prec or box.device != coord.device
        ):
            box = box.to(dtype=self.pt_prec, device=coord.device)

        # requires_grad 必须在 forward_common 之前开启: 描述符由 coord 经
        # extend_input_and_build_neighbor_list 构造的 extended_coord 派生, 若此刻
        # coord 不含梯度, 则 extended_coord/desc 会被当作常量, 长程力会静默丢失电荷
        # 响应项 ∂E_LR/∂q·∂q/∂desc·∂desc/∂r, 只剩 -∂E_LR/∂r|_{q 固定}, 导致能量-力
        # 不自洽。(父类内部对 extended_coord 的 requires_grad_(True) 作用在中间结果上,
        # 是 no-op, 故这里的开关是唯一生效的那次。)
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

        # 2. 调用父类 forward_common 获得短程结果（能量、力等），同时取回复用的描述符。
        #    desc_holder 是「输出通道」: 原子模型算完描述符后把该张量本身 (未 detach)
        #    追加进来, 因此 LES 通道与短程通道共用同一次描述符计算 (省掉一次完整的
        #    descriptor 前向), 且 ∂desc/∂coord 仍保留在计算图中, 长程力的电荷响应项
        #    得以保留。复用的是短程拟合网看到的那一份环境描述符, 二者天然一致;
        #    邻居表的类型区分由父类 format_nlist 按 self.mixed_types() 处理, 因此不再
        #    需要在模型层重建邻居表 (旧版此处硬编码 mixed_types 的行为已随之一并消除)。
        desc_holder = torch.jit.annotate(List[torch.Tensor], [])
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            desc_out=desc_holder,
        )
        assert len(desc_holder) == 1
        desc = desc_holder[0]  # [nframes, nloc, dim]

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
        # type_index: type_map 中的下标 (0..ntypes-1), 供逐类型电荷层
        # (freeze_charge / initial_guess) 索引。与 atomic_numbers
        # (真实原子序数) 不同, 它按 type_map 顺序编号, 因此也能支持非元素符号。
        les_out = self.atomic_model.les_model(
            positions=positions,
            cell=cell_all,
            desc=desc_flat,
            batch=batch,
            compute_energy=True,
            atomic_numbers=atomic_numbers,
            type_index=atype.reshape(-1),
        )
        E_lr_total = les_out["E_lr"]
        # Dict[str, Optional[Tensor]] 返回值精化为 Tensor (TorchScript 需要确定类型)
        assert E_lr_total is not None
        # Ewald 逐图返回标量, torch.cat 之后是一维 [nframes], 而父类的 energy_redu 是
        # [nframes, 1]; 两者直接相加会广播成 [nframes, nframes] 的外和。训练时
        # batch_size auto 取到单帧, 该错误被掩盖 (1x1 矩阵恰好正确), 评估时 batch>1
        # 便得到完全错误的能量矩阵。旧版逐帧循环用 torch.stack 得到的正是 [nframes, 1],
        # 此处显式恢复同一秩, 数值不变 (第 i 行第 i 列即第 i 帧能量)。
        E_lr_total = E_lr_total.reshape(nframes, 1)  # [nframes, 1]
        # 长程能量按 lr_weight 缩放, 与 cace 基准的 CombinePotential mixing weight
        # 对齐。缩放点在梯度计算之前, 因此长程力 (下方对 E_lr_total 求坐标梯度) 与
        # 长程 virial 自动继承同一缩放。
        if self.lr_weight != 1.0:
            E_lr_total = E_lr_total * self.lr_weight

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

    @torch.jit.export
    def need_lower_box(self) -> bool:
        """低层接口 (forward_lower) 是否需要额外传入晶胞。

        Ewald 长程项显式依赖晶胞 (体积、倒格矢、k 空间网格), 而上游的低层接口签名
        不含晶胞: 短程模型可以在 extended 区域上用「力 ⊗ 坐标」还原出 virial, 因此
        不需要晶胞。本模型返回 True, C++ 侧 (api_cc/src/DeepPotPT.cc) 据此决定是否把
        cell 作为第 8 个参数传给 forward_lower; 返回 False 的模型行为与上游一致。
        """
        return True

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
        box: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """低层接口: 输入 extended 区域的坐标/类型/邻居表, 输出未归约到局部原子的量。

        与 forward 的关系: forward = forward_common (内部构建 extended 区域后转调
        forward_common_lower) + LES 长程通道; 本方法直接把 forward_common_lower 暴露
        出来并复用同一个 LES 通道, 因此同一组 extended 输入下二者逐位一致, 长程力/
        virial 的约定 (见 forward 的注释) 也完全沿用。

        box
            晶胞, [nframes, 9], 非周期体系传全零。这是本模型相对上游低层接口的唯一
            扩展参数: 缺少晶胞时长程项无法计算, 故此处显式报错而不是退回只剩短程的
            结果 (静默丢掉长程项会给出错误但看起来正常的能量/力)。

        已知限制 (调用方必须保证):
        长程项是全局量, 要求本帧的局部原子集合构成整个周期体系 (单域运行、串行
        LAMMPS, 或整帧评估)。域分解并行下每个 rank 只持有一部分原子, 长程项需要全局
        电荷归约与倒空间全局求和, 本接口不做 MPI 通信, 因此那种用法下长程项无意义;
        该情形应改用 forward 的整帧通道。
        """
        if box is None:
            raise RuntimeError(
                "HybridLESModel.forward_lower needs the cell to evaluate the "
                "long-range Ewald channel; pass `box` (zeros for a non-periodic "
                "system)."
            )

        # 1. 精度统一, 与 forward 相同: 描述符/LES 参数同在 GLOBAL_PT_FLOAT_PRECISION。
        extended_coord = extended_coord.view(extended_atype.shape[0], -1, 3)
        if extended_coord.dtype != self.pt_prec:
            extended_coord = extended_coord.to(self.pt_prec)
        if box.dtype != self.pt_prec or box.device != extended_coord.device:
            box = box.to(dtype=self.pt_prec, device=extended_coord.device)

        # 2. requires_grad 必须在 forward_common_lower 之前开启, 理由与 forward 中同一段
        #    说明一致: 父类对 extended_coord 的 requires_grad_(True) 作用在中间结果上是
        #    no-op, 而描述符由 extended_coord 派生。此处缺失会同时坏掉两条通道: 短程侧
        #    take_deriv 对 coord_ext 求梯度会直接报错, 长程侧则会静默丢掉电荷响应项。
        if not extended_coord.is_leaf:
            raise RuntimeError(
                "HybridLESModel requires leaf-like extended_coord input to enable "
                "requires_grad_ for the force autograd."
            )
        need_force = self.do_grad_r("energy")
        need_virial = self.do_grad_c("energy")
        need_coord_grad = need_force or need_virial
        if need_coord_grad:
            extended_coord.requires_grad_(True)
        else:
            extended_coord.requires_grad_(False)
        if need_virial:
            box.requires_grad_(True)

        # 3. 短程通道, 同时取回复用的描述符 (输出通道, 见 forward)。
        desc_holder = torch.jit.annotate(List[torch.Tensor], [])
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
            desc_out=desc_holder,
        )
        assert len(desc_holder) == 1
        desc = desc_holder[0]  # [nframes, nloc, dim]

        # 4. LES 长程通道, 与 forward 同构。局部原子是 extended 区域的前 nloc 个 (DeepMD
        #    约定, 邻居表也以它们为中心), 因此 nloc 由邻居表的行数给出, 描述符本身就是
        #    局部原子的 (邻居表只以局部原子为中心), 无需切片。
        nframes = extended_coord.shape[0]
        nloc = nlist.shape[1]
        coord_l = extended_coord[:, :nloc]
        atype_l = extended_atype[:, :nloc]
        batch = (
            torch.arange(nframes, dtype=torch.int64, device=extended_coord.device)
            .repeat_interleave(nloc)
        )
        les_out = self.atomic_model.les_model(
            positions=coord_l.reshape(-1, 3),
            cell=box.reshape(nframes, 3, 3),
            desc=desc.reshape(-1, desc.shape[-1]),
            batch=batch,
            compute_energy=True,
            atomic_numbers=self.atomic_model.element_numbers[
                atype_l.reshape(-1)
            ].to(device=extended_coord.device),
            type_index=atype_l.reshape(-1),
        )
        E_lr_total = les_out["E_lr"]
        assert E_lr_total is not None
        E_lr_total = E_lr_total.reshape(nframes, 1)
        if self.lr_weight != 1.0:
            E_lr_total = E_lr_total * self.lr_weight

        grad_ones = torch.jit.annotate(
            List[Optional[torch.Tensor]], [torch.ones_like(E_lr_total)]
        )
        force_lr_ext: Optional[torch.Tensor] = None
        if need_coord_grad:
            dE_lr_dcoord = torch.autograd.grad(
                [E_lr_total],
                [extended_coord],
                grad_outputs=grad_ones,
                create_graph=True,
                retain_graph=True,
            )[0]  # [nframes, nall, 3]
            assert dE_lr_dcoord is not None
            # E_LR 的 Ewald 求和只接收局部原子位置, 但 ghost 分量并不为零: 局部原子的
            # 描述符依赖 ghost 位置, 电荷经描述符依赖之, 故 dE_LR/d(ghost) != 0。这些
            # 分量按 DeepMD 约定随 extended_force 一并返回, 由调用方用 mapping 归约回
            # 局部原子 (与短程通道完全一致); 下方 virial 的原子项也必须覆盖它们。
            force_lr_ext = -dE_lr_dcoord

        virial_lr: Optional[torch.Tensor] = None
        atom_virial_lr: Optional[torch.Tensor] = None
        if need_virial:
            assert force_lr_ext is not None
            grad_cell = torch.autograd.grad(
                [E_lr_total],
                [box],
                grad_outputs=grad_ones,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            v_cell = torch.zeros(
                nframes, 3, 3, dtype=extended_coord.dtype, device=extended_coord.device
            )
            if grad_cell is not None:
                v_cell = -torch.einsum(
                    "nla,nlb->nab",
                    box.reshape(nframes, 3, 3),
                    grad_cell.reshape(nframes, 3, 3),
                )
            # 与 forward 同一约定 (V = -dE/deps) 与同一指标配对, 见 forward 中的长注释。
            # 短程通道给出的 energy_derv_c_redu 是 extended 区域上的 sum_i F_i (x) r_i,
            # 对周期体系它已经等价于同一应变导数 (ghost 像的位置带出晶胞贡献), 二者可
            # 直接相加。
            #
            # 原子项必须对整个 extended 区域求和, 不能只取前 nloc 个局部原子: ghost 的
            # 长程力非零 (见上), 而像的位置与晶胞成正比 (ext_ghost = wrapped + shift·h),
            # 晶胞对长程 virial 的贡献正是由它们带出来的 —— 这正是短程通道无需显式晶胞项
            # 的同一机制。只对局部原子求和会漏掉该部分: 一帧 192 原子用有限差分对照,
            # 总 virial 偏差达 9.9, 且张量不再对称 (应变反对称部分对应刚体转动, 能量不
            # 变, 其导数必为零, 故对称性可作为此类漏项的哨兵)。
            virial_lr = (
                torch.einsum("nia,nib->nab", extended_coord, force_lr_ext) + v_cell
            ).reshape(nframes, 9)
            # 原子 virial 只保证「逐原子求和 = 总 virial」这一恒等式: 原子项逐 extended
            # 原子给出, 全局的显式晶胞项平均摊到 extended 区域的每个原子上 (与短程通道
            # 把晶胞贡献留在像上的做法同一精神)。
            nall = extended_coord.shape[1]
            atom_virial_lr = (
                torch.einsum("nia,nib->niab", extended_coord, force_lr_ext)
                + (v_cell / nall).unsqueeze(1)
            ).reshape(nframes, nall, 1, 9)

        # 5. 合并结果。输出键与上游 EnergyModel.forward_lower 一致 (C++ 侧读取 energy /
        #    extended_force / virial / extended_virial), 因此这里用 extended 区域的力与
        #    原子 virial, 而不是归约到局部原子后的量。
        model_predict = {}
        model_predict["energy"] = model_ret["energy_redu"] + E_lr_total
        # 长程原子能未定义, 保持短程值 (与 forward 一致)。
        model_predict["atom_energy"] = model_ret["energy"]
        force_sr_ext: Optional[torch.Tensor] = None
        if need_force:
            force_sr_ext = model_ret["energy_derv_r"].squeeze(-2)  # [nframes, nall, 3]
            assert force_lr_ext is not None
            model_predict["extended_force"] = force_sr_ext + force_lr_ext
        if need_virial:
            assert virial_lr is not None
            model_predict["virial"] = (
                model_ret["energy_derv_c_redu"].squeeze(-2) + virial_lr
            )
            if do_atomic_virial:
                assert atom_virial_lr is not None
                model_predict["extended_virial"] = (
                    model_ret["energy_derv_c"].squeeze(-3) + atom_virial_lr
                )

        if torch.jit.is_scripting():
            pass
        else:
            self._maybe_log_les_decomposition(
                model_ret, E_lr_total, force_sr_ext, force_lr_ext
            )
        return model_predict

    @classmethod
    def get_model(cls, model_params: dict) -> "HybridLESModel":
        """Construct a HybridLESModel from a parameter dictionary."""
        # 深拷贝以避免向 model_params 注入 ntypes/type_map/dim_descrpt 等派生键,
        # 污染调用方配置 (影响后续追溯与重复构建)。
        model_params = copy.deepcopy(model_params)
        # les_params 里的 initial_guess / freeze_charge 允许直接传 torch.Tensor,
        # 但下方 model_def_script = json.dumps(model_params) 需要 JSON 可序列化,
        # 故先归一化 (列表 / ndarray 保持不变)。
        model_params["les_params"] = _jsonable(model_params.get("les_params", {}))
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
