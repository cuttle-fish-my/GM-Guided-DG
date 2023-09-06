import argparse
import sys
from copy import deepcopy

import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import InterpolationMode as IM
from torchvision.transforms import functional as F
from tqdm import tqdm

from DatasetProcessor import BaseProcessor

# source: LGE
# target: C0

resize_size = 256
crop_size = 128


class MyoPS2020Processor(BaseProcessor):
    def __init__(self, save_dir, root):
        super().__init__(save_dir, root)

    def load_data(self, i):
        assert i in range(1, 26)
        sources = nib.load(f'MyoPS2020_Raw/train25/myops_training_{i + 100}_DE.nii.gz').get_fdata()
        targets = nib.load(f'MyoPS2020_Raw/train25/myops_training_{i + 100}_C0.nii.gz').get_fdata()
        labels = nib.load(f'MyoPS2020_Raw/train25_myops_gd/myops_training_{i + 100}_gd.nii.gz').get_fdata()

        sources = torch.Tensor(sources.transpose(2, 0, 1))
        targets = torch.Tensor(targets.transpose(2, 0, 1))
        labels = torch.Tensor(labels.transpose(2, 0, 1).astype(np.int32))

        for k, anns in enumerate([0, 200, 500, 600, 1220, 2221]):
            labels[labels == anns] = k
        labels[labels >= 4] = 1  # make scar and edema to be myocardium
        sources = F.resize(sources, [resize_size, resize_size], interpolation=IM.BICUBIC, antialias=False)
        targets = F.resize(targets, [resize_size, resize_size], interpolation=IM.BICUBIC, antialias=False)
        labels = F.resize(labels, [resize_size, resize_size], interpolation=IM.NEAREST, antialias=False)

        sources = F.center_crop(sources, [crop_size, crop_size]).numpy()
        targets = F.center_crop(targets, [crop_size, crop_size]).numpy()
        labels = F.center_crop(labels, [crop_size, crop_size]).numpy()

        sources = sources / np.max(sources, axis=(1, 2), keepdims=True) * 255.0
        targets = targets / np.max(targets, axis=(1, 2), keepdims=True) * 255.0
        return sources, targets, labels, deepcopy(labels)

    def process(self):
        self.makeDir(self.save_dir)
        for i in tqdm(range(1, 26), file=sys.stdout):
            mode = "train" if 1 <= i <= 20 else "val"
            sources, targets, label_sources, label_targets = self.load_data(i)
            self.preprocess(sources, i, self.save_dir, mode, "source")
            self.preprocess(targets, i, self.save_dir, mode, "target")
            self.save_label(label_sources, self.save_dir, mode, "label_source", i)
            self.save_label(label_targets, self.save_dir, mode, "label_target", i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="MyoPS2020")
    args = parser.parse_args()
    processor = MyoPS2020Processor(args.save_dir)
    processor.process()
    # processor.visualize_pair(N=25)
