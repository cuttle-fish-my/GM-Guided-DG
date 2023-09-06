import argparse
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import InterpolationMode as IM
from torchvision.transforms import functional as F

from DatasetProcessor import BaseProcessor

image_size = 128

N = 0


class ProstateProcessor(BaseProcessor):
    """
    Site A: RUNMC
    Site B: BMC
    Site C: I2CVB
    Site D: UCL
    Site E: BIDMC
    Site F: HK
    """

    def __init__(self, save_dir, root, source, target):
        assert source in ["A", "B", "C", "D", "E", "F"]
        assert target in ["A", "B", "C", "D", "E", "F"]
        super().__init__(save_dir, root, source, target, ["BG", "PROSTATE"])
        site_idx = {"A": "RUNMC", "B": "BMC", "C": "I2CVB", "D": "UCL", "E": "BIDMC", "F": "HK"}

        self.source = site_idx[source]
        self.target = site_idx[target]

        self.source_dir = sorted(glob(f"{self.root}/{self.source}/Case*[0123456789].nii.gz"))
        self.label_source_dir = sorted(glob(f"{self.root}/{self.source}/Case*[0123456789]_*egmentation.nii.gz"))

        self.target_dir = sorted(glob(f"{self.root}/{self.target}/Case*[0123456789].nii.gz"))
        self.label_target_dir = sorted(glob(f"{self.root}/{self.target}/Case*[0123456789]_*egmentation.nii.gz"))

    def __len__(self):
        return max(len(self.source_dir), len(self.target_dir))

    def load_data(self, i):
        def load_single_modality(img_dirs, GT_dirs, idx):
            assert len(img_dirs) == len(GT_dirs)
            if idx < 0 or idx >= len(img_dirs):
                return None, None
            img = nib.load(img_dirs[idx]).get_fdata()
            GT = nib.load(GT_dirs[idx]).get_fdata()
            img = torch.Tensor(img.transpose([2, 0, 1]))
            img = F.resize(img, [image_size, image_size], interpolation=IM.BILINEAR, antialias=False)
            GT = torch.Tensor(GT.transpose([2, 0, 1]))
            GT = F.resize(GT, [image_size, image_size], interpolation=IM.NEAREST, antialias=False)
            GT[GT != 0] = 1
            img = img.numpy()
            GT = GT.numpy().astype(np.int32)
            img = img / np.max(img, axis=(1, 2), keepdims=True) * 255.0
            return img, GT
        sources, label_sources = load_single_modality(self.source_dir, self.label_source_dir, i)
        targets, label_targets = load_single_modality(self.target_dir, self.label_target_dir, i)
        return sources, targets, label_sources, label_targets

    def partition_mode(self, i):
        source_mode = "train"
        target_mode = "val"
        return source_mode, target_mode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./Prostate_hist")
    parser.add_argument("--root", type=str, default="./SAML")
    parser.add_argument("--source", type=str, default="A")
    parser.add_argument("--target", type=str, default="E")
    parser.add_argument("--save_meta_info", type=bool, default=True)
    parser.add_argument("--train_source", type=bool, default=False)
    parser.add_argument("--train_target", type=bool, default=False)
    parser.add_argument("--val_source", type=bool, default=False)
    parser.add_argument("--val_target", type=bool, default=True)
    args = parser.parse_args()
    processor = ProstateProcessor(args.save_dir, args.root, args.source, args.target)
    processor.process(args.save_meta_info, equalize_mode="hist",
                      train_source=args.train_source,
                      train_target=args.train_target,
                      val_source=args.val_source,
                      val_target=args.val_target)
