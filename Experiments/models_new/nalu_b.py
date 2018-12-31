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
        #print("INIT_2")
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = 1e-10
        self.initial = ini
        #self.G = Parameter(torch.DoubleTensor(out_dim, in_dim))
        self.nac = NeuralAccumulatorCell(in_dim, out_dim, ini)
        #self.register_parameter('bias', torch.Tensor(out_dim))
        self.bias = Parameter(torch.DoubleTensor(1,out_dim))
        
        if ini =='Kai_uni':
            #init.kaiming_uniform_(self.G, a=math.sqrt(5))
            init.kaiming_uniform_(self.bias)
        if ini =='Xav_norm':
            #init.xavier_normal_(self.G)
            init.xavier_normal_(self.bias)
        if ini =='Kai_norm':
            #init.kaiming_normal_(self.G)
            init.kaiming_normal_(self.bias)
        if ini =='Zeros':
            #init.zeros_(self.G)
            init.zeros_(self.bias)
        if ini =='Ones':
            #init.ones_(self.G)
            init.ones_(self.bias)

    def forward(self, input):
        a = self.nac(input)
        #print(np.shape(self.G))
        #print(np.shape(self.bias))
        g = torch.sigmoid(self.bias)
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
        #print("INIT_1")
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
