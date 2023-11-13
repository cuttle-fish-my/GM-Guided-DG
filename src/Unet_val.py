import argparse
import os
import pathlib
import sys
from datetime import datetime
import json
import PIL.Image
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from GMGADG import utils
from GMGADG.image_datasets import ImageDataset
from GMGADG.losses import entropy_map
from GMGADG.script_util import (
    model_defaults,
    create_model,
    args_to_dict,
    add_dict_to_argparser,
)
from GMGADG.train_util import TestTimeAdaptationAgent, PseudoLabelAgent


def main():
    args = create_argparser().parse_args()
    sub_folder_name = args.model_path.split('/')[-2]
    exp_name = sub_folder_name if args.exp_name == "" else args.exp_name

    if args.TTA_mode is not None:
        exp_name = exp_name + f"_{args.TTA_mode}"

    args.save_dir = os.path.join(args.save_dir, sub_folder_name, exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    dataset = ImageDataset(
        image_paths=args.data_dir,
        modality=args.modality,
        input_mode=args.input_mode,
        mode='val'
    )
    data = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    args.num_class = len(dataset.meta_info['class_list'])

    writer = SummaryWriter(os.path.join(args.save_dir, args.modality, 'logs'))
    print("TTA mode: ", args.TTA_mode)
    if args.TTA_mode == "Entropy":
        model = TestTimeAdaptationAgent(args, writer)
    elif args.TTA_mode == "PseudoLabel":
        model = PseudoLabelAgent(args, writer, dataset.meta_info['class_prior'])
    elif args.TTA_mode is None:
        model = create_model(num_class=args.num_class, **args_to_dict(args, model_defaults().keys()))
        model.load_state_dict(utils.load_state_dict(args.model_path, map_location="cpu"))
        model = model.to(utils.dev())
    else:
        raise NotImplementedError
    model.eval()

    config_str = utils.config_to_html_table(args)
    writer.add_text("config", config_str, 0)

    labels = {}
    preds = {}

    pathlib.Path(os.path.join(args.save_dir, args.modality, 'res')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(args.save_dir, args.modality, 'comparison')).mkdir(parents=True, exist_ok=True)

    for i, (src, out_dict) in enumerate(tqdm(data, file=sys.stdout)):
        file_name = out_dict["file_name"]
        patient_idx, slice_idx, num_slice = file_name[0].split('_')
        if labels.get(patient_idx) is None:
            labels[patient_idx] = [torch.empty(0)] * int(num_slice)
            preds[patient_idx] = [torch.empty(0)] * int(num_slice)
        labels[patient_idx][int(slice_idx) - 1] = out_dict["label"].detach().cpu()

        src = src.to(utils.dev())
        if args.in_channels == 3:
            src = torch.cat([src, src, src], dim=1)

        if args.TTA_mode is not None:
            pred, entropy = model(src, out_dict)
        else:
            pred = model(src)
            pred = torch.softmax(pred, dim=1)
            entropy = entropy_map(pred, reduce=True)

        pred = pred.detach().to('cpu')
        tb_img, metric_string = utils.write_val_img(src, out_dict, pred, writer, i,
                                                     dataset.meta_info['class_list'],
                                                     entropy)

        pred = torch.argmax(pred, dim=1, keepdim=True).cpu()
        preds[patient_idx][int(slice_idx) - 1] = pred
        pred = (pred[0, 0] / args.num_class * 0.8 * 255).numpy().astype(np.uint8)

        if args.save_img:
            PIL.Image.fromarray(pred).save(
                os.path.join(args.save_dir, args.modality, 'res', f'{i}{metric_string}.png'))

            PIL.Image.fromarray(tb_img).save(
                os.path.join(args.save_dir, args.modality, "comparison", f"{i}{metric_string}.png")
            )

    labels = labels.values()
    preds = preds.values()

    label_3d = [torch.stack(label, dim=4) for label in labels]
    pred_3d = [torch.stack(pred, dim=4) for pred in preds]
    metric_dict = utils.CalDice3D(pred_3d, label_3d, dataset.meta_info['class_list'], "dice_3d", writer, False)
    json.dump(metric_dict, open(os.path.join(args.save_dir, args.modality, 'metric.json'), 'w'))


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
        save_img=True,
    )
    defaults.update(model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
