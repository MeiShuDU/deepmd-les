import torch
from torch import nn
from typing import Dict, Any, Union, Optional, List

import logging
logging.basicConfig(
    filename='les.log',
    level=logging.INFO,
    filemode='a',  # append: 保留多轮训练的历史日志, 便于追溯对比
)
logger = logging.getLogger(__name__)

from .module import (
    Atomwise,
    Ewald,
    BEC,
    FixedCharges,
    AtomicAlpha,
    type2number,
)
from .util.scatter import scatter_sum

__all__ = ['Les']


@torch.jit.script
def _broadcast_per_atom(values: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把逐原子量 [n_atoms] 按参考张量的秩升维。

    ref 为 [n_atoms, k] 时返回 [n_atoms, 1], 与逐原子电荷相加时正确广播;
    ref 为一维 (原始 LES 的裸电荷入参) 时原样返回, 避免 [n_atoms,1]+[n_atoms]
    被广播成 [n_atoms, n_atoms] 的静默错误。
    """
    if ref.dim() == 2:
        return values.unsqueeze(-1)
    return values


class Les(nn.Module):

    __constants__ = ['use_fixed_atomic_charges', 'use_atomic_alpha', 'use_atomwise',
                     'use_epsilon_r_scaling', 'is_local_mode', 'is_freeze_mode',
                     'use_initial_guess', 'use_claim_total_charge']
    def __init__(self, les_arguments: Union[Dict[str, Any], str] = {}):
        """
        LES model for long-range interactions

        电荷层 (charge layer) 二选一, 由构造参数决定 (见 _parse_arguments):
        - ``local_charge=True``: 电荷 = Atomwise(描述符), 逐原子、依赖局域环境;
        - ``freeze_charge=[...]``: 电荷 = 逐类型常数, 不参与 SGD, 退化为经典 Ewald。
        逐元素基线 (``use_fixed_charges`` / ``initial_guess``) 是 local 模式的附属,
        每步叠加到 NN 输出上; freeze 模式已直接给定电荷, 不接受基线。
        ``claim_total_charge=S`` 把电荷投影到 ``sum(q) = S`` 的平面上。

        脚本化 (torch.jit.script / dp freeze) 约定:
        - ``compute_energy=True`` 的数值主路径全部由可脚本化算子构成
          (Ewald 已写成 cos/sin 实值形式, 无复数 autograd);
        - 记录/调试分支 (verbose 日志、BEC、atomic_types->type2num 查表) 只在
          eager 下生效, 调用点用 ``if torch.jit.is_scripting(): pass else: ...``
          包住, 且逻辑本体放在 ``@torch.jit.unused`` 方法里 (含 f-string / dict 查表 /
          复数投影等纯 Python 片段的代码不会被编译进计算图); 冻结链路运行时永不进入
          else 分支;
        - FixedCharges/AtomicAlpha 的逐元素查表改为非持久 buffer 张量索引。
        eager 训练路径的行为不变。
        """
        super().__init__()

        if isinstance(les_arguments, str):
            import yaml
            with open(les_arguments, 'r') as file:
                les_arguments = yaml.safe_load(file)
                if les_arguments is None:
                    les_arguments = {}

        self._parse_arguments(les_arguments)

        # 逐类型原子序数 (由 type_map 派生, HybridLESAtomicModel 注入): 非持久
        # buffer, 不进 state_dict, 因此不改变旧 ckpt 的键集合。
        if self._element_numbers is not None:
            self.register_buffer(
                "element_numbers",
                torch.as_tensor(list(self._element_numbers), dtype=torch.int64),
                persistent=False,
            )

        # 注意: 模块属性不能标注泛型 nn.Module (TorchScript 只接受具体模块类型)。
        # 运行时该属性是 Atomwise 或 _DummyAtomwise 的具体实例, 脚本化时按实际
        # 类型推断。hybrid_ener 恒定 use_atomwise=True。
        self.atomwise = (
            Atomwise(
                n_in=self.dim_descrpt,
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                add_linear_nn=self.add_linear_nn,
                output_scaling_factor=self.output_scaling_factor,
            )
            if self.is_local_mode
            else _DummyAtomwise()
        )

        if self.use_fixed_atomic_charges:
            self.fixed_charges = FixedCharges(normalization_factor=self.fixed_atomic_charges_scaling_factor)
        if self.use_atomic_alpha:
            self.atomic_alpha = AtomicAlpha()

        # freeze_charge: 逐类型常数电荷, 不参与 SGD, 框架退化为经典 Ewald。
        # 电荷值完全由构造参数决定, 故用非持久 buffer 承载 (不进 state_dict)。
        if self.is_freeze_mode:
            self.register_buffer(
                "freeze_charge_table",
                self._type_table(self.freeze_charge_values),
                persistent=False,
            )
        # initial_guess: local / 外部电荷模式下逐帧叠加的逐元素基线。
        if self.use_initial_guess:
            self.register_buffer(
                "initial_guess_table",
                self._type_table(self.initial_guess_values),
                persistent=False,
            )

        self.ewald = Ewald(
            sigma=self.sigma,
            dl=self.dl,
            remove_self_interaction=self.remove_self_interaction,
            use_epsilon_r_scaling=self.use_epsilon_r_scaling,
            )

        self.bec = BEC(
             remove_mean=self.remove_mean,
             epsilon_factor=self.epsilon_factor,
             )

    def _parse_arguments(self, les_arguments: Dict[str, Any]):
        """
        Parse arguments for LES model
        """
        self.dim_descrpt = les_arguments.get('dim_descrpt', None)
        self.n_layers = les_arguments.get('n_layers', 3)
        self.n_hidden = les_arguments.get('n_hidden', [32, 16])
        self.add_linear_nn = les_arguments.get('add_linear_nn', True)
        self.output_scaling_factor = les_arguments.get('output_scaling_factor', 0.1)

        self.sigma = les_arguments.get('sigma', 1.0)
        self.dl = les_arguments.get('dl', 2.0)
        self.remove_self_interaction = les_arguments.get('remove_self_interaction', True)

        self.remove_mean = les_arguments.get('remove_mean', True)
        self.epsilon_factor = les_arguments.get('epsilon_factor', 1.)
        # use_fixed_charges 是 use_fixed_atomic_charges 的新名称, 二者等价。
        self.use_fixed_atomic_charges = bool(les_arguments.get(
            'use_fixed_charges',
            les_arguments.get('use_fixed_atomic_charges', False),
        ))
        self.fixed_atomic_charges_scaling_factor = les_arguments.get('fixed_atomic_charges_scaling_factor', 0.5)
        self.use_atomic_alpha = les_arguments.get('use_atomic_alpha', False)
        self.use_epsilon_r_scaling = les_arguments.get('use_epsilon_r_scaling', False)

        # 类型数 / 逐类型原子序数: 由 HybridLESAtomicModel 从 type_map 注入, 供
        # freeze_charge / initial_guess / use_fixed_charges 的逐类型表使用。
        self.ntypes = int(les_arguments.get('ntypes', 0) or 0)
        self._element_numbers = les_arguments.get('element_numbers', None)

        # ---- 电荷层 (charge layer): local / freeze 二选一 ----
        # local_charge 是 use_atomwise 的新名称 (逐原子 NN 拟合, 即既有框架)。
        local_charge = bool(les_arguments.get(
            'local_charge', les_arguments.get('use_atomwise', False)))
        freeze_charge = les_arguments.get('freeze_charge', None)
        n_modes = int(local_charge) + int(freeze_charge is not None)
        if n_modes > 1:
            raise ValueError(
                "local_charge / freeze_charge are mutually exclusive; "
                "enable at most one of them."
            )
        self.is_freeze_mode = freeze_charge is not None
        self.is_local_mode = bool(local_charge) and not self.is_freeze_mode
        # 兼容: use_atomwise 是 local_charge 的历史名称, 外部仍可能读取它。
        self.use_atomwise = self.is_local_mode

        # 逐元素基线附属 (use_fixed_charges / initial_guess): local 模式下每步
        # 叠加到 NN 输出上; freeze 模式已直接给定电荷, 禁止基线。
        self.initial_guess_values = les_arguments.get('initial_guess', None)
        self.freeze_charge_values = freeze_charge
        self.use_initial_guess = self.initial_guess_values is not None

        # ---- claim_total_charge: 把电荷投影到 sum(q) = S 的平面 ----
        # claim_neutral 是 claim_total_charge=0 的简写 (显式电中性条件)。
        claim_neutral = bool(les_arguments.get('claim_neutral', False))
        claim_total_charge = les_arguments.get('claim_total_charge', None)
        if claim_neutral and claim_total_charge is not None:
            raise ValueError(
                "claim_neutral and claim_total_charge are the same switch; set only one."
            )
        if claim_neutral:
            claim_total_charge = 0
        self.use_claim_total_charge = claim_total_charge is not None
        self.claim_total_charge_value = float(claim_total_charge) if claim_total_charge is not None else 0.0

        # ---- 一致性校验 (尽早报错, 避免运行到一半才发现配置冲突) ----
        if (self.is_freeze_mode or self.use_initial_guess) and self.ntypes <= 0:
            raise ValueError(
                "ntypes must be provided (HybridLESAtomicModel injects it from type_map) "
                "when using freeze_charge / initial_guess."
            )
        if self.is_freeze_mode and (self.use_fixed_atomic_charges or self.use_initial_guess):
            raise ValueError(
                "freeze_charge already fixes the charges; it cannot be combined with "
                "use_fixed_charges / initial_guess."
            )

        self.verbose = les_arguments.get('verbose', False)
        if self.verbose:
            self._log_step_counter = 0
            self.log_freq = les_arguments.get('log_freq', 100)
            # log_freq 为非正数时按「关闭日志」处理: 日志里的取模需要正周期,
            # 否则 log_freq=0 会 ZeroDivisionError 而非安静地不打日志。
            if self.log_freq <= 0:
                self.verbose = False

    @torch.jit.unused
    def _type_table(self, values) -> torch.Tensor:
        """把长度 = ntypes 的 list / Tensor 规范成 [ntypes] float64 张量。

        纯 Python 分支 (逐元素 float / 长度校验), 只在 __init__ (eager) 中调用。
        """
        if torch.is_tensor(values):
            table = values.detach().reshape(-1).to(torch.float64)
        else:
            table = torch.tensor([float(v) for v in values], dtype=torch.float64)
        if table.numel() != self.ntypes:
            raise ValueError(
                f"expected {self.ntypes} values (one per type_map entry), "
                f"got {table.numel()}."
            )
        return table

    def __setstate__(self, state: Dict[str, Any]):
        # Backward compatibility: models serialized before these feature flags
        # existed lack the corresponding attributes. forward() and __constants__
        # now access them directly (no hasattr), so we restore their historical
        # defaults at deserialization time. This keeps forward() TorchScript-
        # friendly while still loading (and scripting) older checkpoints.
        # 电荷层开关是后加的: 旧 ckpt 只有 use_atomwise, 对应 local 模式。
        for key, default in {
            'use_atomwise': False,
            'use_fixed_atomic_charges': False,
            'use_atomic_alpha': False,
            'use_epsilon_r_scaling': False,
            'is_local_mode': bool(state.get('use_atomwise', False)),
            'is_freeze_mode': False,
            'use_initial_guess': False,
            'use_claim_total_charge': False,
            'claim_total_charge_value': 0.0,
            'ntypes': 0,
        }.items():
            state.setdefault(key, default)
        super().__setstate__(state)

    @torch.jit.unused
    def _atomic_numbers_from_types(self, atomic_types: list) -> torch.Tensor:
        # pure Python dict 查表 (type2number), 只在 eager 下被调用。
        # dp freeze 链路应直接传 atomic_numbers (int64 张量)。
        return type2number.type2num(atomic_types)

    @torch.jit.unused
    def _log_verbose_stats(self,
                           latent_charges: Optional[torch.Tensor],
                           atomic_numbers: Optional[torch.Tensor],
                           E_lr: Optional[torch.Tensor],
                           compute_energy: bool,
                           ) -> None:
        # 日志/调试分支: 含 f-string 格式、逐元素统计、E_lr.grad_fn 等纯 Python
        # 片段, 由 @torch.jit.unused 隔离, 不进入冻结模型的计算图。
        if not self.verbose:
            return
        self._log_step_counter += 1
        if self._log_step_counter % self.log_freq == 0 or self._log_step_counter == 1:
            logger.info(f'Training steps :{self._log_step_counter}')
            if latent_charges is not None:
                logger.info(f"latent_charges mean: {latent_charges.mean().item()}, std: {latent_charges.std().item()}")
                if atomic_numbers is not None:
                    logger.info(f"\tQ_MEAN\tQ_STD")
                    for am in torch.unique(atomic_numbers):
                        mask = (atomic_numbers == am)
                        q_am = latent_charges[mask]
                        mean_q = q_am.mean(dim=0)
                        std_q = q_am.std(dim=0)
                        logger.info(f"{am}\t{mean_q}\t{std_q}")
                else:
                    logger.info("atomic_numbers is None; skip per-element Q stats")
            if compute_energy:
                logger.info(f"E_lr = {E_lr.item() if E_lr.numel()==1 else E_lr}")
        if self._log_step_counter % self.log_freq == 0 or self._log_step_counter == 1:
            logger.info("Q layer params (norm summary):")
            for name, param in self.named_parameters():
                pnorm = param.detach().norm().item() if param.numel() else 0.0
                frozen = "frozen" if not param.requires_grad else "train"
                logger.info(f"  {name}: |w|={pnorm:.6e} ({frozen})")
            if compute_energy:
                logger.info(f"E_lr graph : {E_lr.grad_fn}")

    def forward(self,
               positions: torch.Tensor, # [n_atoms, 3]
               cell: Optional[torch.Tensor] = None, # [batch_size, 3, 3] (无盒子时传 None, 走实空间直接求和)
               e_ext: Optional[torch.Tensor]= None,
               desc: Optional[torch.Tensor]= None, # [n_atoms, n_features]
               latent_charges: Optional[torch.Tensor] = None, # [n_atoms, ]
               latent_dipoles: Optional[torch.Tensor] = None, # [n_atoms, 3]
               latent_quads: Optional[torch.Tensor] = None, # [n_atoms, 3, 3]
               latent_kappas: Optional[torch.Tensor] = None, # [n_atoms, ]
               latent_alphas: Optional[torch.Tensor] = None, # [n_atoms, ]
               atomic_numbers: Optional[torch.Tensor] = None, # [n_atoms, ]
               atomic_types: Optional[List[str]] = None,
               type_index: Optional[torch.Tensor] = None, # [n_atoms, ] type_map 中的下标 (0..ntypes-1)
               batch: Optional[torch.Tensor] = None,
               compute_energy: bool = True,
               compute_field: bool = False,
               compute_bec: bool = False,
               bec_output_index: Optional[int] = None, # option to compute BEC components along only one direction
               ) -> Dict[str, Optional[torch.Tensor]]:
        """
        arguments:
        desc: torch.Tensor
        Descriptors for the atoms. Shape: (n_atoms, n_features)
        latent_charges: torch.Tensor
        One can also directly input the latent charges. Shape: (n_atoms, )
        positions: torch.Tensor
            positions of the atoms. Shape: (n_atoms, 3)
        cell: torch.Tensor
            cell of the system. Shape: (batch_size, 3, 3)
        batch: torch.Tensor
            batch of the system. Shape: (n_atoms,)
        """
        # check the input shapes
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.int64, device=positions.device)

        if type_index is not None:
            type_index = type_index.to(device=positions.device, dtype=torch.int64)

        # 电荷来源: 外部传入的 latent_charges 优先 (raw LES / BEC 通道); 否则按构造
        # 时选定的电荷层产生 (local=逐原子 NN, freeze=逐类型常数)。
        if latent_charges is not None:
            # check the shape of latent charges
            assert latent_charges.shape[0] == positions.shape[0]
        elif self.is_freeze_mode:
            assert type_index is not None, "freeze_charge mode requires type_index"
            latent_charges = self.freeze_charge_table[type_index].unsqueeze(-1)
        elif self.is_local_mode:
            if desc is None:
                raise ValueError("desc must be provided when latent_charges is not provided")
            # compute the latent charges
            assert desc.shape[0] == positions.shape[0]
            latent_charges = self.atomwise(desc, batch)
        else:
            raise ValueError("Either desc or latent_charges must be provided")
        # Optional[Tensor] 精化为 Tensor: TorchScript 需要后续的 q 入参类型确定
        assert latent_charges is not None

        # atomic_types->atomic_numbers 走 Python dict 查表 (type2number), 逻辑本体在
        # @torch.jit.unused 方法中, 编译期不参与; 冻结链路直接传 atomic_numbers。
        if torch.jit.is_scripting():
            pass
        else:
            if atomic_types is not None:
                atomic_numbers = self._atomic_numbers_from_types(atomic_types)

        # 固定电荷/原子极化率查表是模块内 buffer, 索引张量需与 buffer 同设备。
        if atomic_numbers is not None:
            atomic_numbers = atomic_numbers.to(device=latent_charges.device)

        # 逐元素基线: 叠加在电荷来源之上 (freeze 模式禁止基线, 见 _parse_arguments)。
        if atomic_numbers is not None and self.use_fixed_atomic_charges:
            latent_charges = latent_charges + _broadcast_per_atom(
                self.fixed_charges(atomic_numbers), latent_charges)
        if type_index is not None and self.use_initial_guess:
            latent_charges = latent_charges + _broadcast_per_atom(
                self.initial_guess_table[type_index], latent_charges)

        # claim_total_charge: 把电荷投影到 sum(q) = S 的平面, 逐帧独立 (每个 frame
        # 是一个独立的周期体系)。投影是线性幂等映射, autograd 直接穿过, 因此 q 每步
        # 都精确满足约束, 同时仍被 SGD 更新 —— 即 GUIDANCE 三种处置方案中的方案 1
        # 「直接训练 q, 不关心其自身取值」, 约束由前向投影精确保证而非靠惩罚项逼近。
        # 方案 2 (omega*(Q-S)^2 正则) 需要把惩罚项送进 deepmd 的 loss, 会让 eval 的
        # 能量带上偏移; 方案 3 (只更新 q_r 再回写 q) 与方案 1 在数学上等价 (投影矩阵
        # 幂等), 却多一份状态管理。故取方案 1。
        if self.use_claim_total_charge:
            nframes = int(batch.max().item()) + 1
            counts = torch.bincount(batch, minlength=nframes).to(latent_charges.dtype)
            counts = _broadcast_per_atom(counts, latent_charges)
            q_sum = scatter_sum(latent_charges, batch, 0, None, nframes)
            excess = (q_sum - self.claim_total_charge_value) / counts
            latent_charges = latent_charges - excess[batch]

        if atomic_numbers is not None and self.use_atomic_alpha and latent_alphas is not None:
            baseline_alphas = self.atomic_alpha(atomic_numbers)
            if latent_alphas.dim() == 1:
                latent_alphas = latent_alphas + baseline_alphas
            elif latent_alphas.dim() == 3:
                latent_alphas = latent_alphas + baseline_alphas[:,None,None] * torch.eye(3, device=baseline_alphas.device).unsqueeze(0) # [n_atoms, 3, 3]

        # compute the long-range interactions
        if compute_energy:
            E_lr, q_induced, u_induced = self.ewald(q=latent_charges,
                              u=latent_dipoles,
                              kappa=latent_kappas,
                              alpha=latent_alphas,
                              quad=latent_quads,
                              r=positions,
                              cell=cell,
                              batch=batch,
                              compute_field=compute_field,
                              e_ext = e_ext,
                              )
        else:
            E_lr, q_induced, u_induced = None, None, None

        if latent_alphas is not None and u_induced is not None:
            if latent_dipoles is not None:
                if latent_dipoles.dim() == 2 and u_induced.dim() == 3:
                    latent_dipoles = latent_dipoles.unsqueeze(1) # [n_node, 1, 3]
                assert latent_dipoles.shape == u_induced.shape, f'latent_dipoles dimension error'
                latent_dipoles = latent_dipoles + u_induced
            else:
                latent_dipoles = u_induced

        if latent_kappas is not None and q_induced is not None:
            latent_charges = latent_charges + q_induced

        # BEC 仅被显式请求时计算, 由 @torch.jit.unused 的 BEC.forward 承担 (内含
        # 复数投影与高阶 autograd), 不进入冻结模型的计算图。
        bec: Optional[torch.Tensor] = None
        if torch.jit.is_scripting():
            pass
        else:
            if compute_bec:
                bec = self.bec(q=latent_charges,
                               u=latent_dipoles,
                               r=positions,
                               cell=cell,
                               batch=batch,
                               output_index=bec_output_index,
                               )

        # 日志/调试分支仅 eager 下生效 (见 _log_verbose_stats)。
        if torch.jit.is_scripting():
            pass
        else:
            self._log_verbose_stats(latent_charges, atomic_numbers, E_lr, compute_energy)

        output = {
            'E_lr': E_lr,
            'latent_charges': latent_charges,
            'latent_dipoles': latent_dipoles,
            'latent_quads': latent_quads,
            'latent_alphas': latent_alphas,
            'BEC': bec,
            }
        return output

class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        raise ValueError("set use_atomwise to True to use Atomwise module")