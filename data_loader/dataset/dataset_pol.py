from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from utils import util, torch_funcs


# To align with the settings of pCON paper, we adopt similar preprocessing approaches.
# The only difference is the convention about Malus' law.
# You can also set pCON_convention=True and re-train the network to keep the same as pCON paper
class PCONDataset(Dataset):
    def __init__(self, data_dir, scene_name, img_size, num_points, pCON_convention=False):
        base_path = Path(data_dir) / scene_name
        path_list = sorted(list(base_path.glob('*.png')), key=lambda path: int(path.stem.split('_')[-1]))
        print(base_path)
        self.I1_path, self.I2_path, self.I3_path, self.I4_path = path_list

        # (2048 2448 C) float32 ndarray --> (2048 2048 C) float32 ndarray
        I1 = util.read_img(self.I1_path, rgb=True)[:, 200:-200, :]
        I2 = util.read_img(self.I2_path, rgb=True)[:, 200:-200, :]
        I3 = util.read_img(self.I3_path, rgb=True)[:, 200:-200, :]
        I4 = util.read_img(self.I4_path, rgb=True)[:, 200:-200, :]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
        ])

        # (3 H W) float32 tensor
        self.I1 = transform(I1)
        self.I2 = transform(I2)
        self.I3 = transform(I3)
        self.I4 = transform(I4)

        # torch.linspace(-1, 1, x): from -1 to 1 (-1 and 1 are included as the start and end), get x points in total
        self.xy = torch.stack(
            torch.meshgrid(torch.linspace(-1, 1, img_size[0]), torch.linspace(-1, 1, img_size[1]), indexing='ij'),
            dim=0
        )  # (2 H W)

        # calculate necessary variables
        if pCON_convention:
            # using the convention used by pCON paper
            self.S0 = (self.I1 + self.I2 + self.I3 + self.I4) / 2
            self.S1 = 2 * self.I1 - self.S0
            self.S2 = 2 * self.I2 - self.S0
            self.p = torch.sqrt(self.S1 ** 2 + self.S2 ** 2) / (self.S0 + 1e-10)
            self.theta = (0.5 * torch.atan2(self.S2, self.S1 + 1e-10)) % np.pi
            a = self.S1 / (self.S0 + 1e-10)
            b = self.S2 / (self.S0 + 1e-10)
            self.k = a + 1j * b
        else:
            # using our convention
            self.S0, self.S1, self.S2 = torch_funcs.imgs2Stokes(self.I1, self.I2, self.I3, self.I4)  # (3 H W), real
            self.p, self.theta = torch_funcs.Stokes2DoPAoP(self.S0, self.S1, self.S2)  # (3 H W), real
            self.k = torch_funcs.Stokes2k(self.S0, self.S1, self.S2)  # (3 H W), complex

        # flatten
        G = (img_size[0] * img_size[1]) // num_points
        self.flattened_xy = self.xy.permute(1, 2, 0).reshape(G, num_points, 2)  # (G num_points 2) real
        self.flattened_S0 = self.S0.permute(1, 2, 0).reshape(G, num_points, 3)  # (G num_points 3) real
        self.flattened_k = self.k.permute(1, 2, 0).reshape(G, num_points, 3)  # (G num_points 3) complex
        self.flattened_p = self.p.permute(1, 2, 0).reshape(G, num_points, 3)  # (G num_points 3) complex
        self.flattened_theta = self.theta.permute(1, 2, 0).reshape(G, num_points, 3)  # (G num_points 3) complex

        self.name = scene_name

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            'xy': self.xy,
            'S0': self.S0,
            'S1': self.S1,
            'S2': self.S2,
            'p': self.p,
            'theta': self.theta,
            'k': self.k,
            'flattened_xy': self.flattened_xy,
            'flattened_S0': self.flattened_S0,
            'flattened_k': self.flattened_k,
            'flattened_p': self.flattened_p,
            'flattened_theta': self.flattened_theta,
            'name': self.name
        }
