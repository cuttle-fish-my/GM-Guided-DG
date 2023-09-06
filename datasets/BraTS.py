import argparse
import os
from copy import deepcopy
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import InterpolationMode as IM
from torchvision.transforms import functional as F

from datasets.DatasetProcessor import BaseProcessor

resize_size = 128


class BraTSProcessor(BaseProcessor):
    def __init__(self, save_dir, root, source, target):
        assert source in ["flair", "t1", "t1ce", "t2"]
        assert target in ["flair", "t1", "t1ce", "t2"]
        super().__init__(save_dir, root, source, target, ["BG", "Tumor"])

        self.val_list = sorted(glob(f"{self.root}/HGG/Brats18*"))[-21:] + sorted(glob(f"{self.root}/LGG/Brats18*"))[-7:]

        self.folder_list = sorted(glob(f"{self.root}/*/Brats18*"))

        self.source_dir = sorted(glob(f"{self.root}/*/Brats18*/*{source}.nii.gz"))
        self.target_dir = sorted(glob(f"{self.root}/*/Brats18*/*{target}.nii.gz"))
        self.label_dir = sorted(glob(f"{self.root}/*/Brats18*/*seg.nii.gz"))

    def __len__(self):
        return len(self.folder_list)

    @staticmethod
    def _BBox(img_1, img_2):
        x_1, y_1, z_1 = np.where(img_1.astype(np.int32) != 0)
        x_2, y_2, z_2 = np.where(img_2.astype(np.int32) != 0)

        x_min = min(np.min(x_1), np.min(x_2))
        x_max = max(np.max(x_1), np.max(x_2))
        y_min = min(np.min(y_1), np.min(y_2))
        y_max = max(np.max(y_1), np.max(y_2))
        z_min = min(np.min(z_1), np.min(z_2))
        z_max = max(np.max(z_1), np.max(z_2))

        return x_min, x_max, y_min, y_max, z_min, z_max

    def load_data(self, i):
        sources = nib.load(self.source_dir[i]).get_fdata()
        targets = nib.load(self.target_dir[i]).get_fdata()
        label = nib.load(self.label_dir[i]).get_fdata()

        label[label != 0] = 1

        sources = np.transpose(sources, (2, 0, 1))
        targets = np.transpose(targets, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        x_min, x_max, y_min, y_max, z_min, z_max = self._BBox(sources, targets)

        sources = torch.Tensor(sources[x_min:x_max, y_min:y_max, z_min:z_max])
        targets = torch.Tensor(targets[x_min:x_max, y_min:y_max, z_min:z_max])
        label = torch.Tensor(label[x_min:x_max, y_min:y_max, z_min:z_max])

        sources = F.resize(sources, [resize_size, resize_size], IM.BILINEAR, antialias=False).numpy()
        targets = F.resize(targets, [resize_size, resize_size], IM.BILINEAR, antialias=False).numpy()
        label = F.resize(label, [resize_size, resize_size], IM.NEAREST, antialias=False).numpy().astype(np.int32)

        sources = sources / (np.max(sources, axis=(1, 2), keepdims=True) + 1e-8) * 255.0
        targets = targets / (np.max(targets, axis=(1, 2), keepdims=True) + 1e-8) * 255.0

        return sources, targets, label, deepcopy(label)

    def partition_mode(self, i):
        assert os.path.dirname(self.source_dir[i]) == self.folder_list[i]
        mode = "val" if self.folder_list[i] in self.val_list else "train"
        return mode, deepcopy(mode)

    def BBOX(self, i):
        sources = nib.load(self.source_dir[i]).get_fdata()
        origin_shape = sources.shape
        targets = nib.load(self.target_dir[i]).get_fdata()
        label = nib.load(self.label_dir[i]).get_fdata()

        label[label != 0] = 1

        sources = np.transpose(sources, (2, 0, 1))
        targets = np.transpose(targets, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        x_min, x_max, y_min, y_max, z_min, z_max = self._BBox(sources, targets)
        return y_min, y_max, z_min, z_max, origin_shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./BraTS")
    parser.add_argument("--root", type=str, default="./BraTS_Raw")
    parser.add_argument("--source", type=str, default="t2")
    parser.add_argument("--target", type=str, default="t1ce")
    parser.add_argument("--save_meta_info", type=bool, default=True)
    parser.add_argument("--train_source", type=bool, default=False)
    parser.add_argument("--train_target", type=bool, default=False)
    parser.add_argument("--val_source", type=bool, default=False)
    parser.add_argument("--val_target", type=bool, default=True)
    args = parser.parse_args()
    processor = BraTSProcessor(args.save_dir, args.root, source=args.source, target=args.target)
    processor.process(save_meta_info=args.save_meta_info,
                      train_source=args.train_source,
                      train_target=args.train_target,
                      val_source=args.val_source,
                      val_target=args.val_target)
