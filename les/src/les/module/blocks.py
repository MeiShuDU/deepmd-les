from typing import Callable, Union, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["build_mlp", "Dense"]


def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Optional[nn.Module] = None,
    bias: bool = True,
) -> nn.Module:
    """
    Build multiple layer fully connected perceptron neural network.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: 激活函数模块 (nn.Module)。None 表示默认使用 SiLU。
            注意: 这里必须是 nn.Module 而非裸函数 (如 torch.nn.functional.silu),
            因为 torch.jit.script 要求子模块是具体模块类型, 裸函数无法作为
            模块属性被脚本化。SiLU 与旧实现使用的 F.silu 数值完全一致。
    """
    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # get list of number of nodes hidden layers
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        n_neurons = [n_in] + n_hidden + [n_out]

    act = nn.SiLU() if activation is None else activation

    # assign a Dense layer (with activation function) to each hidden layer
    layers = [
        Dense(n_neurons[i], n_neurons[i + 1], activation=act, bias=bias)
        for i in range(n_layers - 1)
    ]

    # assign a Dense layer (without activation function) to the output layer
    layers.append(
        Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=bias)
    )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net

class Dense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[nn.Module] = None,
    ):
        """
        Fully connected linear layer with an optional activation function.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If False, the layer will not have a bias term.
            activation (nn.Module): 激活函数模块。None 表示恒等 (nn.Identity)。
        """
        super().__init__()
        # Dense layer
        self.linear = nn.Linear(in_features, out_features, bias)

        # Activation function
        self.activation = nn.Identity() if activation is None else activation

    def forward(self, input: torch.Tensor):
        y = self.linear(input)
        y = self.activation(y)
        return y
