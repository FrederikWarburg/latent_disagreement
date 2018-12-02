import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .nac import NeuralAccumulatorCell
from torch.nn.parameter import Parameter
import numpy as np

class NeuralArithmeticLogicUnitCell(nn.Module):
    """A Neural Arithmetic Logic Unit (NALU) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/abs/1808.00508
    """
    def __init__(self, in_dim, out_dim, ini):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = 1e-10
        self.initial = ini
        self.G = Parameter(torch.Tensor(out_dim, in_dim))
        self.nac = NeuralAccumulatorCell(in_dim, out_dim, ini)
        self.register_parameter('bias', None)

        if ini =='Kai_uni':
            init.kaiming_uniform_(self.G, a=math.sqrt(5))

        if ini =='Xav_norm':
            init.xavier_normal_(self.G)

        if ini =='Kai_norm':
            init.kaiming_normal_(self.G)

        if ini =='Zeros':
            init.zeros_(self.G)

        if ini =='Ones':
            init.ones_(self.G)


    def forward(self, input):
        a = self.nac(input)
        g = torch.sigmoid(F.linear(input, self.G, self.bias))
        self.g_store = g
        add_sub = g * a
        log_input = torch.log(torch.abs(input) + self.eps)
        m = torch.exp(self.nac(log_input))
        mul_div = (1 - g) * m
        y = add_sub + mul_div
        return y

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(
            self.in_dim, self.out_dim
        )


class NALU(nn.Module):
    """A stack of NAC layers.

    Attributes:
        num_layers: the number of NAC layers.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
    """
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, ini):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.initial = ini
        layers = []
        for i in range(num_layers):
            layers.append(
                NeuralArithmeticLogicUnitCell(hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < num_layers - 1 else out_dim, ini))
                    
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out
