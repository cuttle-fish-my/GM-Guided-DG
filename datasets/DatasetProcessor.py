import pathlib
import pickle
import sys
from abc import abstractmethod, ABC

import numpy as np
import skimage
from scipy import ndimage as ndi
from skimage import filters as filters
from tqdm import tqdm


class BaseProcessor(ABC):
    def __init__(self, save_dir, root, source, target, class_list):
        """
        :param save_dir: str, path to save the processed data
        :param root: str, path to the raw data
        :param source: str, source modality
        :param target: str, target modality
        :param class_list: list, list of classes. Including BG, e.g. ["BG", "class1", "class2, ..., "classN"]
        """
        self.root = root
        self.source = source
        self.target = target
        self.save_dir = save_dir + f"_{source}2{target}"
        self.class_list = class_list

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def load_data(self, i):
        pass

    def process(self, save_meta_info=False, equalize_mode="hist",
                train_source=True, train_target=True, val_source=True, val_target=True):
        self.makeDir(self.save_dir)
        if save_meta_info:
            self.save_meta_info()
        for i in tqdm(range(self.__len__()), desc="process", file=sys.stdout):
            source_mode, target_mode = self.partition_mode(i)
            sources, targets, label_sources, label_targets = self.load_data(i)
            if train_source and source_mode == "train" and sources is not None:
                self.preprocess(sources, i + 1, self.save_dir, source_mode, "source", equalize_mode=equalize_mode)
                self.save_label(label_sources, self.save_dir, source_mode, "label_source", i + 1)
            if train_target and target_mode == "train" and targets is not None:
                self.preprocess(targets, i + 1, self.save_dir, target_mode, "target", equalize_mode=equalize_mode)
                self.save_label(label_targets, self.save_dir, target_mode, "label_target", i + 1)
            if val_source and source_mode == "val" and sources is not None:
                self.preprocess(sources, i + 1, self.save_dir, source_mode, "source", equalize_mode=equalize_mode)
                self.save_label(label_sources, self.save_dir, source_mode, "label_source", i + 1)
            if val_target and target_mode == "val" and targets is not None:
                self.preprocess(targets, i + 1, self.save_dir, target_mode, "target", equalize_mode=equalize_mode)
                self.save_label(label_targets, self.save_dir, target_mode, "label_target", i + 1)

    @staticmethod
    def makeDir(Dir):
        for mode in ["train", "val"]:
            for name in ["source", "target", "label_source", "label_target"]:
                pathlib.Path(f"{Dir}/{mode}/{name}").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_label(labels, Dir, mode, name, i):
        total = labels.shape[0]
        for j in range(labels.shape[0]):
            label = labels[j]
            np.save(f"{Dir}/{mode}/{name}/{i}_{j + 1}_{total}.npy", label[None])

    def save_meta_info(self):
        meta_info = {
            "class_list": self.class_list,
            "class_prior": self.class_prior_statistic(),
        }
        pickle.dump(meta_info, open(f"{self.save_dir}/train/meta_info.pkl", "wb"))
        pickle.dump(meta_info, open(f"{self.save_dir}/val/meta_info.pkl", "wb"))

    @staticmethod
    def preprocess(imgs, i=None, Dir=None, mode=None, name=None, save=True, equalize_mode="hist"):
        assert mode in ["train", "val", None]
        assert name in ["source", "target", None]
        new_imgs = []
        for j in range(imgs.shape[0]):
            img = imgs[j, ...]
            origin = img
            img = filters.gaussian(img, sigma=0.5)
            g_x = ndi.correlate(img, np.array([[-1, 0, 1]]), mode="reflect")
            g_y = ndi.correlate(img, np.array([[-1], [0], [1]]), mode="reflect")
            magnitude = np.sqrt(g_x ** 2 + g_y ** 2)
            if equalize_mode == "hist":
                mask = magnitude > 0
                if mask.any():
                    magnitude = skimage.exposure.equalize_hist(BaseProcessor.rescale(magnitude), mask=mask)
                magnitude *= mask
                magnitude *= 255
            elif equalize_mode == "ada_hist":
                magnitude = skimage.exposure.equalize_adapthist(BaseProcessor.rescale(magnitude)) * 255
            elif equalize_mode == "minmax":
                magnitude = BaseProcessor.rescale(magnitude) * 255
            else:
                raise NotImplementedError
            img = np.stack([origin, magnitude], axis=0)
            new_imgs.append(img)

            if save:
                np.save(f"{Dir}/{mode}/{name}/{i}_{j + 1}_{imgs.shape[0]}.npy", img)
        return np.array(new_imgs)

    @staticmethod
    def rescale(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    def visualize_pair(self, N=None):
        from torch.utils.tensorboard import SummaryWriter
        from einops import rearrange
        if N is None:
            N = self.__len__()
        writer = SummaryWriter('pairs/logs')
        for i in tqdm(range(N), desc="Visualizing pairs"):
            sources, targets, label_sources, label_targets = self.load_data(i)
            matched_imgs = []
            matched_labels = []
            for j in range(sources.shape[0]):
                matched_imgs.append(targets[int(round(j / (sources.shape[0] - 1) * (targets.shape[0] - 1)))])
                matched_labels.append(label_targets[int(round(j / (sources.shape[0] - 1) * (targets.shape[0] - 1)))])
            matched_imgs = np.stack(matched_imgs)
            matched_labels = np.stack(matched_labels)
            tb_img = np.stack([sources, matched_imgs], axis=0)
            tb_img = rearrange(tb_img, 'r c h w -> (r h) (c w)')
            tb_img /= 255
            writer.add_image(f"MRI/patient_{i}", tb_img, dataformats='HW')
            tb_label = np.stack([label_sources, matched_labels], axis=0)
            tb_label = rearrange(tb_label, 'r c h w -> (r h) (c w)') / 3 * 0.8
            writer.add_image(f"Label/patient_{i}_label", tb_label, dataformats='HW')

    def visualize_origin_magnitude_angle(self, N=None):
        from torch.utils.tensorboard import SummaryWriter
        from einops import rearrange
        if N is None:
            N = self.__len__()
        writer = SummaryWriter('origin_magnitude_angle/logs')
        for i in tqdm(range(N), desc="Visualizing origin, magnitude and angle"):
            sources, targets, _, _ = self.load_data(i)
            src_mao = self.preprocess(sources, save=False) / 255
            tgt_mao = self.preprocess(targets, save=False) / 255
            tb_src_img = rearrange(src_mao, "n c h w -> (n h) (c w)")
            tb_tgt_img = rearrange(tgt_mao, "n c h w -> (n h) (c w)")
            writer.add_image(f"src/{i}", tb_src_img, dataformats='HW')
            writer.add_image(f"tgt/{i}", tb_tgt_img, dataformats='HW')

    def class_prior_statistic(self, N=None):
        if N is None:
            N = self.__len__()
        class_prior = np.zeros(len(self.class_list))
        for i in tqdm(range(N), desc="class prior statistic", file=sys.stdout):
            if self.partition_mode(i)[0] != "train":
                continue
            label_source = self.load_data(i)[-2]
            if label_source is not None:
                class_prior += np.eye(len(self.class_list))[label_source.flatten()].sum(axis=0)
        class_prior /= np.sum(class_prior)
        return class_prior

    @abstractmethod
    def partition_mode(self, i):
        pass
