import os
import pickle
import random

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

import imgaug as ia
import imgaug.augmenters as iaa


def load_data(
        *,
        dataset,
        batch_size,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.
    :param dataset: the dataset to load.
    :param batch_size: the batch size of each returned pair.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(self,
                 image_paths,
                 modality='source',
                 input_mode='magnitude',
                 mode="train",
                 heavy_aug=False,
                 ):
        super().__init__()
        print(f"modality: {modality}, input_mode: {input_mode}, mode: {mode}")
        assert modality in ['source', 'target']
        assert input_mode in ['origin', 'magnitude']
        assert mode in ["train", "val"]
        self.mode = mode
        self.modality = modality
        input_mode_idx = {'origin': [0], 'magnitude': [1]}
        self.input_mode_idx = input_mode_idx[input_mode]
        self.heavy_aug = heavy_aug

        self.img_path = os.path.join(image_paths, modality)
        self.label_path = os.path.join(image_paths, 'label_' + modality)

        self.img_files = sorted(os.listdir(self.img_path))
        self.label_files = sorted(os.listdir(self.label_path))

        self.meta_info = pickle.load(open(os.path.join(image_paths, 'meta_info.pkl'), 'rb'))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        img = np.load(os.path.join(self.img_path, file_name)) / 127.5 - 1
        out_dict = {
            'origin': img[0][None],
        }

        label = np.load(os.path.join(self.label_path, self.label_files[idx]))
        img = img[self.input_mode_idx]

        if self.mode == "train":
            if self.heavy_aug:
                img = ((np.repeat(img[..., None], 3, axis=-1) + 1) * 127.5).astype(np.uint8)
                label = label.astype(np.uint8)
                img, label = augmentation(img, label[0][None, ..., None])
                img = img[0].astype(np.float32) / 127.5 - 1
                img = np.transpose(img, [2, 0, 1])
                label = torch.tensor(label[0].transpose([2, 0, 1]))
            else:
                img, label = random_affine_transform([img, label])
        else:
            img = torch.Tensor(img)
            label = torch.Tensor(label)
        out_dict.update({
            'label': label.long(),
            'name': self.img_files[idx],
            'file_name': file_name.split('.')[0],
        })

        return img, out_dict


class UnpairedDataset(Dataset):
    def __init__(self,
                 image_paths,
                 mode="train",
                 modality='source',
                 input_mode="origin",
                 heavy_aug=False,
                 ):
        super().__init__()
        print(f"input_mode: {input_mode}, mode: {mode}")

        assert mode in ["train", "val"], f"mode should be train or val, but got {mode}"
        assert modality in ['source', 'target'], f"modality should be source or target, but got {modality}"
        assert input_mode in ['origin', 'magnitude',
                              'angle'], f"input_mode should in origin, magnitude, angle, but got {input_mode}"
        self.mode = mode
        self.input_mode_idx = {'origin': 0, 'magnitude': 1, 'angle': 2}[input_mode]
        self.heavy_aug = heavy_aug
        self.source_path = os.path.join(image_paths, 'target' if modality == 'source' else 'source')
        self.target_path = os.path.join(image_paths, modality)
        self.label_path = os.path.join(image_paths, f'label_{modality}')
        self.source_files = sorted(os.listdir(self.source_path))
        self.target_files = sorted(os.listdir(self.target_path))
        self.label_files = sorted(os.listdir(self.label_path))
        self.meta_info = pickle.load(open(os.path.join(image_paths, 'meta_info.pkl'), 'rb'))

    def __len__(self):
        if self.mode == "val":
            return len(self.source_files)
        return len(self.target_files)

    def __getitem__(self, idx):
        source_idx = random.randint(0, len(self.source_path) - 1)
        # source = np.load(os.path.join(self.source_path, self.source_files[source_idx])) / 127.5 - 1
        # target = np.load(os.path.join(self.target_path, self.target_files[idx])) / 127.5 - 1
        source = np.load(os.path.join(self.source_path, self.source_files[source_idx])).astype(np.uint8)
        out_dict = {
            'origin': source[0][None].astype(np.float32) / 127.5 - 1,
            'magnitude': source[1][None].astype(np.float32) / 127.5 - 1,
            'angle': source[2][None].astype(np.float32) / 127.5 - 1,
        }
        target = np.load(os.path.join(self.target_path, self.target_files[idx])).astype(np.uint8)
        label = np.load(os.path.join(self.label_path, self.label_files[idx]))
        if self.mode == "train":
            if self.heavy_aug:
                target = np.repeat(target[self.input_mode_idx][..., None], 3, axis=2)
                target, label = augmentation(target[None], label[0][None, ..., None])
                source = np.repeat(source[self.input_mode_idx][..., None], 3, axis=2)
                source = augmentation(source[None], None)[0]

                target = target[0].transpose([2, 0, 1])
                source = source[0].transpose([2, 0, 1])
                label = label[0].transpose([2, 0, 1])
            else:
                source = random_affine_transform(source[0][None])
                target, label = random_affine_transform([target[0][None], label])
                source = torch.cat([source, source, source], dim=0).numpy()
                target = torch.cat([target, target, target], dim=0).numpy()
                label = label.numpy()
        target = target.astype(np.float32) / 127.5 - 1
        source = source.astype(np.float32) / 127.5 - 1
        out_dict.update({
            "source": source,
            "label": label.astype(np.int64),
        })
        return target, out_dict


def random_affine_transform(imgs, random_rotate=True, random_shift=True, random_scale=True, ):
    if not isinstance(imgs, list):
        imgs = [imgs]
    channels = [img.shape[0] for img in imgs]
    img = np.concatenate(imgs, axis=0)
    img = torch.tensor(img, dtype=torch.float32)
    affine = T.RandomAffine(
        degrees=20 if random_rotate else 0,
        translate=(0.35, 0.35) if random_shift else None,
        scale=(0.9, 1.1) if random_scale else None,
    )
    img = affine(img)
    imgs = torch.split(img, channels, dim=0)
    return imgs if len(imgs) > 1 else imgs[0]


def augmentation(image, mask):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 11)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # iaa.SimplexNoiseAlpha(iaa.OneOf([
                           #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                           #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           # ])),
                           iaa.BlendAlphaSimplexNoise(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           # add gaussian noise to images
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           ]),
                           iaa.Invert(0.05, per_channel=True),  # invert color channels
                           iaa.Add((-10, 10), per_channel=0.5),
                           iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                           iaa.OneOf([
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           ]),
                           iaa.Grayscale(alpha=(0.0, 1.0)),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )
    image_heavy, mask_heavy = seq(images=image, segmentation_maps=mask)
    return image_heavy, mask_heavy
