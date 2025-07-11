import torch
import torch.nn as nn
import numpy as np

from base.base_model import BaseModel
from .blocks.block_utils.Complex import cLinear, cSineLayer


class DefaultModel(BaseModel):
    def __init__(self, hidden_features=256, hidden_layers=10, out_features=3, first_omega_0=60, hidden_omega_0=30,
                 sin_type='cSin_cartesian', affinity=True):
        super(DefaultModel, self).__init__()

        self.wave_encoder = cSineLayer(2, hidden_features, bias=True, is_first=True, omega_0=first_omega_0,
                                       sin_type=sin_type)
        self.message_encoder = nn.Sequential(
            *[cSineLayer(hidden_features, hidden_features, bias=True, is_first=False, omega_0=hidden_omega_0,
                         sin_type=sin_type) for _ in range(hidden_layers // 3)]
        )
        self.carrier_encoder = cSineLayer(hidden_features, hidden_features, bias=True, is_first=False,
                                          omega_0=hidden_omega_0, sin_type=sin_type)
        self.k_encoder = nn.Sequential(
            *[cSineLayer(hidden_features, hidden_features, bias=True, is_first=False, omega_0=hidden_omega_0,
                         sin_type=sin_type) for _ in range(hidden_layers // 3)]
        )
        self.theta_encoder = nn.Sequential(
            *[cSineLayer(hidden_features, hidden_features, bias=True, is_first=False, omega_0=hidden_omega_0,
                         sin_type=sin_type) for _ in range(hidden_layers // 3)]
        )

        self.message_decoder = nn.Linear(2 * hidden_features, out_features, bias=True)
        self.k_decoder = cLinear(hidden_features, out_features, bias=True)
        self.theta_decoder = nn.Linear(4 * hidden_features, out_features, bias=True)

        with torch.no_grad():
            self.message_decoder.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                                 np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.k_decoder.weight.real.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                                np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.k_decoder.weight.imag.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                                np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.theta_decoder.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                               np.sqrt(6 / hidden_features) / hidden_omega_0)

        self.affinity = affinity

    def c2r(self, t):
        if self.affinity:
            # [real1, imag1, real2, imag2, ...]
            B, _, N_ch, C_ch = t.shape
            t_combined = torch.stack(
                (t.real, t.imag), dim=-1
            ).view(B, _, N_ch, 2 * C_ch)
        else:
            # [real1, real2, ..., imag1, imag2, ...]
            t_combined = torch.cat((t.real, t.imag), dim=-1)
        return t_combined

    def forward(self, xy):
        # network
        wave_features = self.wave_encoder(xy)
        message_features = self.message_encoder(wave_features)
        carrier_features = self.carrier_encoder(wave_features)
        k_features = self.k_encoder(carrier_features)
        theta_features = self.theta_encoder(carrier_features)

        S0_pred = self.message_decoder(self.c2r(message_features))
        S0_pred = torch.clamp(S0_pred, min=0, max=1)

        k_pred = self.k_decoder(k_features)
        k_pred = torch.polar(torch.clamp(k_pred.abs(), min=0, max=1), k_pred.angle())

        theta_pred = self.theta_decoder(torch.cat((self.c2r(theta_features), self.c2r(k_features)), dim=-1))
        theta_pred = torch.clamp(theta_pred, min=0, max=1)

        return S0_pred, k_pred, theta_pred
