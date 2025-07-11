from .dataset import dataset_pol
from base.base_data_loader import BaseDataLoader


class PCONDataLoader(BaseDataLoader):
    def __init__(self, data_dir, scene_name, img_size=(1024, 1024), num_points=524288, pCON_convention=False):
        self.img_size = img_size
        self.num_points = num_points
        self.pCON_convention = pCON_convention
        self.dataset = dataset_pol.PCONDataset(data_dir, scene_name, self.img_size, self.num_points,
                                               self.pCON_convention)

        assert ((self.img_size[0] * self.img_size[0]) % self.num_points == 0), "num_points not evenly divisible"

        super(PCONDataLoader, self).__init__(self.dataset, batch_size=1, shuffle=False, validation_split=0.0,
                                             num_workers=0)
