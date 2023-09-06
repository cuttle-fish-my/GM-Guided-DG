import argparse
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import InterpolationMode as IM
from torchvision.transforms import functional as F

from DatasetProcessor import BaseProcessor

crop_size = 128


class MSCMRSeg2019Processor(BaseProcessor):
    def __init__(self, save_dir, root, source, target, source_len=None):
        self.modalities = ["C0", "LGE", "T2"]
        self.train_len = {"C0": 35, "LGE": 5, "T2": 35}
        if source_len is not None:
            self.train_len[source] = source_len
        assert source in self.modalities and target in self.modalities
        super().__init__(save_dir, root, source, target, ["BG", "Myo", "LV", "RV"])

        img_dirs = {}
        GT_dirs = {}

        for modality in [self.source, self.target]:
            img_dirs[modality] = sorted(glob(f"{root}/*/*/patient*[0123456789]_{modality}.nii.gz"),
                                        key=lambda x: int(x.split("/")[-1].split("_")[0].split("patient")[1]))
            GT_dirs[modality] = sorted(glob(f"{root}/*/*/patient*[0123456789]_{modality}_manual.nii.gz"),
                                       key=lambda x: int(x.split("/")[-1].split("_")[0].split("patient")[1]))

        self.source_dir = img_dirs[source]
        self.target_dir = img_dirs[target]
        self.label_source_dir = GT_dirs[source]
        self.label_target_dir = GT_dirs[target]

    def __len__(self):
        return len(self.source_dir)

    def load_data(self, i):
        sources = nib.load(self.source_dir[i]).get_fdata()
        targets = nib.load(self.target_dir[i]).get_fdata()
        label_sources = nib.load(self.label_source_dir[i]).get_fdata()
        label_targets = nib.load(self.label_target_dir[i]).get_fdata()

        sources = torch.Tensor(sources.transpose([2, 0, 1]))
        targets = torch.Tensor(targets.transpose([2, 0, 1]))
        label_targets = torch.Tensor(label_targets.transpose([2, 0, 1]))
        label_sources = torch.Tensor(label_sources.transpose([2, 0, 1]))

        targets = F.resize(targets, sources.shape[1:], antialias=False)
        label_targets = F.resize(label_targets, sources.shape[1:], interpolation=IM.NEAREST, antialias=False)

        sources = F.center_crop(sources, [crop_size, crop_size])
        targets = F.center_crop(targets, [crop_size, crop_size])
        label_targets = F.center_crop(label_targets, [crop_size, crop_size])
        label_sources = F.center_crop(label_sources, [crop_size, crop_size])

        sources = sources.numpy()
        targets = targets.numpy()
        label_targets = label_targets.numpy().astype(np.int32)
        label_sources = label_sources.numpy().astype(np.int32)

        for idx, v in enumerate([0, 200, 500, 600]):
            label_targets[label_targets == v] = idx
            label_sources[label_sources == v] = idx

        sources = sources / np.max(sources, axis=(1, 2), keepdims=True) * 255.0
        targets = targets / np.max(targets, axis=(1, 2), keepdims=True) * 255.0
        return sources, targets, label_sources, label_targets

    def partition_mode(self, i):
        i += 1
        source_mode = "train" if 1 <= i <= self.train_len[self.source] else "val"
        target_mode = "train" if 1 <= i <= self.train_len[self.target] else "val"
        return source_mode, target_mode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="MS-CMRSeg2019")
    parser.add_argument("--root", type=str, default="MS-CMRSeg2019_Raw")
    parser.add_argument("--source", type=str, default="C0")
    parser.add_argument("--target", type=str, default="T2")
    parser.add_argument("--save_meta_info", type=bool, default=True)
    parser.add_argument("--train_source", type=bool, default=True)
    parser.add_argument("--train_target", type=bool, default=False)
    parser.add_argument("--val_source", type=bool, default=False)
    parser.add_argument("--val_target", type=bool, default=True)
    parser.add_argument("--source_num", type=int, default=35)
    args = parser.parse_args()
    processor = MSCMRSeg2019Processor(args.save_dir, args.root, args.source, args.target, args.source_num)
    processor.process(save_meta_info=args.save_meta_info,
                      train_source=args.train_source,
                      train_target=args.train_target,
                      val_source=args.val_source,
                      val_target=args.val_target)
    # processor.save_meta_info()
    # processor.visualize_pair()
    # processor.visualize_origin_magnitude_angle()
