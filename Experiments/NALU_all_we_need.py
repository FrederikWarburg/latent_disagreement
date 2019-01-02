import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from models.nac import NeuralAccumulatorCell
from torch.nn.parameter import Parameter

import random
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class NeuralAccumulatorCell(nn.Module):
    """A Neural Accumulator (NAC) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/abs/1808.00508
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = 0

        self.W_hat = Parameter(torch.Tensor(out_dim, in_dim))
        self.M_hat = Parameter(torch.Tensor(out_dim, in_dim))

        self.register_parameter('W_hat', self.W_hat)
        self.register_parameter('M_hat', self.M_hat)
        self.register_parameter('bias', None)

        self._reset_params()

    def _reset_params(self):
        init.kaiming_uniform_(self.W_hat)
        init.kaiming_uniform_(self.M_hat)

    def forward(self, input):
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        self.W = W
        return F.linear(input, W, self.bias)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(
            self.in_dim, self.out_dim
        )


class NAC(nn.Module):
    """A stack of NAC layers.

    Attributes:
        num_layers: the number of NAC layers.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
    """
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        layers = []
        for i in range(num_layers):
            layers.append(
                NeuralAccumulatorCell(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < num_layers - 1 else out_dim,
                )
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out

    
class NeuralArithmeticLogicUnitCell(nn.Module):
    """A Neural Arithmetic Logic Unit (NALU) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/abs/1808.00508
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = 1e-10

        self.G = Parameter(torch.Tensor(out_dim, in_dim))
        self.nac = NeuralAccumulatorCell(in_dim, out_dim)
        self.register_parameter('bias', None)

        init.kaiming_uniform_(self.G, a=math.sqrt(5))

    def forward(self, input):

        a = self.nac(input)
        g = torch.sigmoid(F.linear(input, self.G, self.bias))
        add_sub = g * a
        log_input = torch.log(torch.abs(input) + self.eps)
        m = torch.exp(self.nac(log_input))
        mul_div = (1 - g) * m
        y = add_sub + mul_div
        return y, g, self.nac.W, self.G

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
    
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.g = []
        self.W = []
        self.G = []

        layers = []
        for i in range(num_layers):
            layers.append(
                NeuralArithmeticLogicUnitCell(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < num_layers - 1 else out_dim,
                )
            )
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, training):

        out, g , W, G = self.model(x)
        
        if training:
        
            self.W.append(W.data.numpy())
            self.g.append(g)
            self.G.append(G.detach().numpy().copy())
        
        return out
    
def test(model, data, target):
    with torch.no_grad():
        out = model(data)
        return torch.abs(target - out)
    
    
def train(model, optimizer, num_iters, train_support, val_support, input_noise, target_noise):
    training_loss = []
    training_error = []
    val_error = []
    train_data = []
    val_data = []

    for i in range(num_iters):
        
        # Training
        training = True
        x_train, y_train = generate_data(fn, train_support, input_noise, target_noise)
        train_data.append(y_train.detach().numpy().copy())
        
        out = model(x_train, training)

        loss = F.mse_loss(out, y_train)
        training_loss.append(loss.data.numpy())
        
        training_error.append( (y_train - out).detach().numpy().copy() )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        training = False
        x_val, y_val = generate_data(fn, val_support, input_noise, target_noise)
        val_data.append(y_val.detach().numpy().copy())

        out = model(x_val, training)
        
        val_error.append( (y_val - out).detach().numpy().copy() )
        
        if i % 1000 == 0:
            acc = np.sum(np.isclose(output_batch, y_batch, atol=.1, rtol=0)) / len(y_batch)
            print('epoch {2}, loss: {0}, accuracy: {1}'.format(l, acc, epoch))


    return training_loss, training_error, val_error, train_data, val_data
            