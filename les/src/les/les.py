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

__all__ = ['Les']

class Les(nn.Module):

    __constants__ = ['use_fixed_atomic_charges', 'use_atomic_alpha', 'use_atomwise', 'use_epsilon_r_scaling']
    def __init__(self, les_arguments: Union[Dict[str, Any], str] = {}):
        """
        LES model for long-range interactions

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
            if self.use_atomwise
            else _DummyAtomwise()
        )

        if self.use_fixed_atomic_charges:
            self.fixed_charges = FixedCharges(normalization_factor=self.fixed_atomic_charges_scaling_factor)
        if self.use_atomic_alpha:
            self.atomic_alpha = AtomicAlpha()

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
        self.use_atomwise = les_arguments.get('use_atomwise', False)
        self.use_fixed_atomic_charges = les_arguments.get('use_fixed_atomic_charges', False)
        self.fixed_atomic_charges_scaling_factor = les_arguments.get('fixed_atomic_charges_scaling_factor', 0.5)
        self.use_atomic_alpha = les_arguments.get('use_atomic_alpha', False)
        self.use_epsilon_r_scaling = les_arguments.get('use_epsilon_r_scaling', False)

        self.verbose = les_arguments.get('verbose', False)
        if self.verbose:
            self._log_step_counter = 0
            self.log_freq = les_arguments.get('log_freq', 100)

    def __setstate__(self, state: Dict[str, Any]):
        # Backward compatibility: models serialized before these feature flags
        # existed lack the corresponding attributes. forward() and __constants__
        # now access them directly (no hasattr), so we restore their historical
        # defaults at deserialization time. This keeps forward() TorchScript-
        # friendly while still loading (and scripting) older checkpoints.
        for key, default in {
            'use_atomwise': False,
            'use_fixed_atomic_charges': False,
            'use_atomic_alpha': False,
            'use_epsilon_r_scaling': False,
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
            # 权重 norm 摘要。注意: 该打印发生在 forward 内, 而 DeePMD 训练
            # 循环在每步 forward 之前已调用 optimizer.zero_grad(), 因此此处
            # 读取 param.grad 永远为 None (无意义); 梯度是否到达 LES 应通过
            # HybridLESModel 注册的 backward hook (打印 [LES-grad]) 或在
            # loss.backward() 之后观察。这里只记录权重演化以追溯学习。
            logger.info("Q NN params (norm summary):")
            for name, param in self.atomwise.named_parameters():
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

        if latent_charges is not None:
            # check the shape of latent charges
            assert latent_charges.shape[0] == positions.shape[0]
        elif desc is not None and latent_charges is None:
            if not self.use_atomwise:
                raise ValueError("desc must be provided and use_atomwise must be True if latent_charges is not provided")
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

        if atomic_numbers is not None and self.use_fixed_atomic_charges:
            latent_charges = latent_charges + self.fixed_charges(atomic_numbers).unsqueeze(-1)

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