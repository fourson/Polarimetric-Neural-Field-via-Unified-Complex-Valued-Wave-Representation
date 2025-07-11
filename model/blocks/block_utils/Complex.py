import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class cLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(cLinear, self).__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.zeros(1, out_features, dtype=torch.cfloat), requires_grad=bias)

        # nn.init.xavier_uniform_(self.weight)
        # if bias:
        #     nn.init.zeros_(self.bias)

    def forward(self, inp):
        if not inp.dtype == torch.cfloat:
            inp = torch.complex(inp, torch.zeros_like(inp))
        return torch.matmul(inp, self.weight.T) + self.bias


def cSin_cartesian(inp):
    sin_real = torch.sin(inp.real)
    sin_imag = torch.sin(inp.imag)
    return sin_real + 1j * sin_imag


def cSin_polar(inp):
    sin_abs = torch.sin(inp.abs())
    sin_angle = torch.sin(inp.angle())
    return torch.polar(sin_abs, sin_angle)


class cSineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30., sin_type='cSin_cartesian'):
        super(cSineLayer, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.sin_type = sin_type

        self.in_features = in_features
        self.clinear = cLinear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.clinear.weight.real.uniform_(-1 / self.in_features, 1 / self.in_features)
                self.clinear.weight.imag.uniform_(-1 / self.in_features, 1 / self.in_features)
                # self.clinear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.clinear.weight.real.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                                  np.sqrt(6 / self.in_features) / self.omega_0)
                self.clinear.weight.imag.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                                  np.sqrt(6 / self.in_features) / self.omega_0)
                # self.clinear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                #                              np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        temp = self.omega_0 * self.clinear(x)
        if self.sin_type == 'cSin_cartesian':
            out = cSin_cartesian(temp)
        elif self.sin_type == 'cSin_polar':
            out = cSin_polar(temp)
        else:
            raise NotImplementedError(f'{self.sin_type} not exists!')
        return out


def complexGelu(inp):
    return torch.complex(F.gelu(inp.real), F.gelu(inp.imag))


class cGelu(nn.Module):
    @staticmethod
    def forward(inp):
        return complexGelu(inp)


class cMLP(nn.Module):
    """
    Complex Multilayer Perceptron
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(cMLP, self).__init__()
        self.num_layers = num_layers
        self.input_layer = cLinear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([cLinear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = cLinear(hidden_size, output_size)
        self.activation = cGelu()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for i in range(self.num_layers - 1):
            x = self.activation(self.hidden_layers[i](x))
        output = self.output_layer(x)
        return output
