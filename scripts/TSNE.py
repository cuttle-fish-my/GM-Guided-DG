import argparse
import os
from copy import copy

import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn
from matplotlib import pyplot as plt
from sklearn import manifold
from torch.utils.tensorboard import SummaryWriter

from guided_diffusion import utils
from guided_diffusion.image_datasets import ImageDataset
from guided_diffusion.script_util import model_defaults, add_dict_to_argparser
from guided_diffusion.train_util import TSNEAgent

sns.set(rc={'figure.figsize': (10, 10)})


def main():
    args = create_argparser().parse_args()
    sub_folder_name = args.model_path.split('/')[-2]
    exp_name = sub_folder_name if args.exp_name == "" else args.exp_name
    exp_name += "TSNE"
    args.save_dir = os.path.join(args.save_dir, sub_folder_name, exp_name, str(args.selected_idx))

    writer = SummaryWriter(os.path.join(args.save_dir, args.modality, 'logs'))

    dataset = ImageDataset(
        image_paths=args.data_dir,
        modality=args.modality,
        input_mode=args.input_mode,
        mode='val'
    )

    args.num_class = len(dataset.meta_info['class_list'])

    model = TSNEAgent(args, writer, dataset.meta_info['class_prior'])

    config_str = utils.config_to_html_table(args)
    writer.add_text("config", config_str, 0)

    src, out_dict = dataset[args.selected_idx]
    src = src[None]
    out_dict["label"] = out_dict["label"][None]
    label = out_dict["label"].flatten().cpu().detach().numpy()

    src = src.to(utils.dev())
    src = torch.cat([src, src, src], dim=1)
    preds, features = model(src, out_dict)
    #     visualize the features via TSNE
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for idx, (pred, feature) in enumerate(zip(preds, features)):
        if not os.path.exists(os.path.join(args.save_dir, args.modality, f'tsne_{idx}.csv')):
            mask = np.logical_xor(pred == 1, label == 1)
            new_label = copy(label)
            new_label[mask] = 4

            X_tsne = tsne.fit_transform(feature)
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_tsne = (X_tsne - x_min) / (x_max - x_min)
            df = pd.DataFrame(X_tsne, columns=['x', 'y'])

            df['label'] = new_label
            # sort df by label
            df = df.sort_values(by=['label'])

            string_label = ["BG"] * len(new_label)
            for s, v in {"BG": 0, "Myo": 1, "LV": 2, "RV": 3, "XOR": 4}.items():
                for i in np.where(df['label'] == v)[0].tolist():
                    string_label[i] = s
            df['label'] = string_label
            # save df to disk
            df.to_csv(os.path.join(args.save_dir, args.modality, f'tsne_{idx}.csv'), index=False)
        else:
            df = pd.read_csv(os.path.join(args.save_dir, args.modality, f'tsne_{idx}.csv'))
        palette = [(1.0, 0.498, 0.0549),
                   (0.122, 0.467, 0.706),
                   (0.173, 0.627, 0.173),
                   (0.580, 0.404, 0.741),
                   (0.839, 0.153, 0.157),
                   ]

        ax = sns.scatterplot(data=df, x='x', y='y', hue='label',
                             s=200, ax=axes[idx],
                             hue_order=['BG', 'Myo', 'LV', 'RV', 'XOR'],
                             style='label', style_order=['BG', 'Myo', 'LV', 'RV', 'XOR'],
                             edgecolor='black', linewidth=0.2, alpha=0.8,
                             palette=palette)
        ax.set(xticklabels=[], yticklabels=[])
        ax.set(xlabel=None, ylabel=None)
        plt.setp(ax.get_legend().get_texts(), fontsize='20')
        plt.setp(ax.get_legend().get_title(), fontsize='30')
    plt.show()


def create_argparser():
    defaults = dict(
        data_dir="",
        batch_size=1,
        model_path="",
        modality='source',
        input_mode='magnitude',
        save_dir="./val_res",
        exp_name="",
        TTA_mode=None,
        TTA_lr=1e-3,
        TTA_steps=1,
        TTA_episodic=False,
        TTA_alpha=0.9,
        TTA_class_idx=-1,
        lambda_BN=0.6,
        lambda_ent=2,
        lambda_consistency=0.1,
        selected_idx=0,
    )
    defaults.update(model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
