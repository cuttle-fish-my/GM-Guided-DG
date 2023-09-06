import argparse
import os
import sys

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from universeg import universeg

from guided_diffusion import utils
from guided_diffusion import utils
from guided_diffusion.image_datasets import ImageDataset


class SupportSet(Dataset):
    def __init__(self, Dir, support_len=10, seed=None):
        self.img_dir = os.path.join(Dir, 'train', 'image')
        self.label_dir = os.path.join(Dir, 'train', 'label')
        self.img_files = os.listdir(self.img_dir)
        self.label_files = os.listdir(self.label_dir)
        self.img_files.sort()
        self.label_files.sort()
        self.support_len = support_len
        if seed is not None:
            np.random.seed(seed)
        self.support_idx = np.random.choice(len(self.img_files), self.support_len, replace=False)

    def _process_img(self, Dir):
        img = np.load(Dir)
        img = img.astype(np.float32) / 255
        return img

    def get_support_set(self, fixed=True):
        support_imgs = []
        support_labels = []
        support_idx = self.support_idx if fixed else np.random.choice(len(self.img_files), self.support_len,
                                                                      replace=False)
        for i in support_idx:
            support_imgs.append(self._process_img(os.path.join(self.img_dir, self.img_files[i])))
            support_label = np.load(os.path.join(self.label_dir, self.label_files[i]))
            support_label = support_label.astype(np.float32)
            support_labels.append(support_label)
        support_imgs = np.stack(support_imgs)[:, None]
        support_labels = np.stack(support_labels)[:, None]
        support_labels = (support_labels == 5).astype(np.int32)  # we only care about scars
        return torch.tensor(support_imgs), torch.tensor(support_labels)


@torch.no_grad()
def main(opts):
    model = universeg(pretrained=True).to(utils.dev())
    model.eval()
    writer = SummaryWriter(opts.save_dir)
    dataset = ImageDataset(
        image_paths=opts.data_dir,
        modality="source",
        input_mode="origin",
        mode="val",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    support_set = SupportSet(opts.support_dir, support_len=opts.support_len)
    # support_images, support_labels = support_set.get_support_set()
    # support_img = torch.cat([support_images, support_labels], dim=1)
    # support_img = rearrange(support_img, 'n b h w -> (b h) (n w)')
    # writer.add_images("support_set", support_img, dataformats='HW')
    for idx, (image, _) in enumerate(tqdm(loader, desc="evaluating", position=0, file=sys.stdout)):
        tb_img = []
        image = (image + 1) / 2
        preds = []
        for i in tqdm(range(args.ensemble_size), desc="ensemble", position=1, file=sys.stdout):
            support_images, support_labels = support_set.get_support_set(fixed=False)
            preds.append(model(image.to(utils.dev()),
                               support_images[None].to(utils.dev()),
                               support_labels[None].to(utils.dev())))
        pred = torch.mean(torch.stack(preds), dim=0).to('cpu')
        tb_img.append(image)
        tb_img.append(pred)
        tb_img.append(pred > 0.5)
        tb_img.append(utils.apply_mask(pred, image))
        for i in range(len(tb_img)):
            if tb_img[i].shape[1] == 1:
                tb_img[i] = tb_img[i].repeat(1, 3, 1, 1)
        tb_img = torch.stack(tb_img, dim=1)
        tb_img = rearrange(tb_img, 'b n c h w -> c (b h) (n w)')
        writer.add_images(f"scar_pred/{idx}", tb_img, dataformats='CHW')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/MS-CMRSeg2019/val", help="path to dataset")
    parser.add_argument("--support_dir", type=str, default="../datasets/MyoPS2020/", help="path to support set")
    parser.add_argument("--save_dir", type=str, default="UniverSeg/logs", help="path to save results")
    parser.add_argument("--support_len", type=int, default=10, help="number of support images")
    parser.add_argument("--ensemble_size", type=int, default=5, help="number of ensemble models")
    args = parser.parse_args()
    main(args)
