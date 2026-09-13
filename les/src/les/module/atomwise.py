from typing import Dict, Union, Sequence, Optional

import torch
import torch.nn as nn
from .blocks import Dense, build_mlp
#from ..util import scatter_sum

__all__ = ["Atomwise"]

class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    脚本化 (torch.jit.script) 约定:
    - ``n_in`` 必须在构造时给出 (``Les`` 从 ``les_params['dim_descrpt']`` 传入),
      这样 ``self.outnet`` 在 ``__init__`` 内即为具体模块, forward 中不再有
      「outnet is None 时惰性构建」的分支;
    - ``activation`` 为 nn.Module (默认 nn.SiLU, 与 F.silu 数值一致)。
    """

    def __init__(
        self,
        n_in: Optional[int] = None,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        bias: bool = True,
        activation: Optional[nn.Module] = None,
        add_linear_nn: bool = False,
        output_scaling_factor: float = 1.0,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            add_linear_nn: whether to add a linear NN to the output of the MLP
        """
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = nn.SiLU() if activation is None else activation
        self.add_linear_nn = add_linear_nn
        self.bias = bias
        self.output_scaling_factor = output_scaling_factor
        if n_in is not None:
            self._build(n_in)

    @torch.jit.unused
    def _build(self, n_in:int):
        # 惰性构建只在 __init__ (未传 n_in) 或 eager forward 中使用。脚本化编译时
        # 不编译本方法 (内含 build_mlp 的模块构建过程), 冻结链路要求 n_in 在
        # __init__ 中已给出, self.outnet 恒为具体模块。
        self.n_in = n_in
        self.outnet = build_mlp(
            n_in=n_in, n_out=self.n_out, n_hidden=self.n_hidden,
            n_layers=self.n_layers, activation=self.activation, bias=self.bias,
        )
        if self.add_linear_nn:
            self.linear_nn = Dense(n_in, self.n_out, bias=self.bias, activation=None)
        else: self.linear_nn = None

    def forward(self,
                desc: torch.Tensor, # [n_atoms, n_features]
                batch: torch.Tensor, # [n_atoms]
                training: bool = None,
               ) -> torch.Tensor:

        # 惰性构建仅在 eager 下可用 (供未传 n_in 的调用方使用); 脚本化编译时该
        # 分支不生效, self.outnet 始终是 __init__ 中构建好的具体 nn.Sequential。
        if torch.jit.is_scripting():
            pass
        else:
            if self.n_in is None:
                self._build(desc.shape[1])
        assert self.n_in is not None and self.n_in == desc.shape[1], "Atomwise: n_in mismatch"

        # predict atomwise contributions
        y = self.outnet(desc)
        if self.add_linear_nn:
            y = y + self.linear_nn(desc)

        return y * self.output_scaling_factor

    def __repr__(self):
        return f"Atomwise(n_in={self.n_in}, n_out={self.n_out}, n_hidden={self.n_hidden}, n_layers={self.n_layers}, bias={self.bias}, activation={type(self.activation).__name__})"

