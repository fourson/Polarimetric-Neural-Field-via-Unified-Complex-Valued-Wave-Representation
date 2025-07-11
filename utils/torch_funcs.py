import numpy as np
import torch


def imgs2Stokes(I1, I2, I3, I4):
    S0 = (I1 + I2 + I3 + I4) / 2
    S1 = I3 - I1
    S2 = I4 - I2
    return S0, S1, S2


def Stokes2DoPAoP(S0, S1, S2):
    p = torch.clamp(torch.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-10), min=0, max=1)  # in [0, 1]
    theta = (torch.atan2(S2, S1) / 2) % np.pi  # in [0, pi]
    return p, theta


def Stokes2k(S0, S1, S2):
    a = S1 / (S0 + 1e-10)
    b = S2 / (S0 + 1e-10)
    return a + 1j * b


def k2DoPAoP(k):
    p = k.abs()  # in [0, 1]
    theta = (k.angle() / 2) % np.pi  # in [0, pi]
    return p, theta


def Stokes2imgs(S0, S1, S2):
    I1 = (S0 - S1) / 2
    I2 = (S0 - S2) / 2
    I3 = (S0 + S1) / 2
    I4 = (S0 + S2) / 2
    return I1, I2, I3, I4


def Stokes2img(S0, S1, S2, alpha):
    I_alpha = (S0 - torch.cos(2 * alpha) * S1 - torch.sin(2 * alpha) * S2) / 2
    I_alpha = torch.clamp(I_alpha, min=0, max=1)  # in [0, 1]
    return I_alpha
