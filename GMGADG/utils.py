import io
import platform
import sys
from typing import List

import blobfile as bf
import numpy as np
import torch
from einops import rearrange
from monai.metrics import DiceMetric
from torch.utils.tensorboard import SummaryWriter

COLORS = torch.tensor(
    [[0, 1, 1],
     [1, 0, 1],
     [1, 1, 0],
     [0.3, 0.7, 0.2],
     [0.5, 0.2, 0.7]]
)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda")
    elif "macOS" in platform.platform() and "arm64" in platform.platform() and getattr(sys, 'gettrace', None)() is None:
        return torch.device("mps")
    return torch.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), **kwargs)


def CalDice3D(prediction: List[torch.Tensor],  # List[Batch, Channel, Height, Width, Slices]
              groundTruth: List[torch.Tensor],  # List[Batch, Channel, Height, Width, Slices]
              class_list: List[str],
              metric_name: str = "dice_3d",
              writer: SummaryWriter = None,
              reference_result: bool = False):
    Dices = []
    ref_Dices = []
    for pred, gt in zip(prediction, groundTruth):
        dice_dict = CalDice(pred, gt, class_list, writer=None, reference_result=reference_result)
        Dices.append(dice_dict["dice"])
        ref_Dices.append(dice_dict["ref_dice"]) if reference_result else None
    Dices = torch.Tensor(Dices).mean(dim=0).tolist()
    metric_dict = write_metric(writer, Dices, metric_name, class_list)
    if reference_result:
        ref_Dices = torch.Tensor(ref_Dices).mean(dim=0).tolist()
        write_metric(writer, ref_Dices, "reference_" + metric_name, class_list) if reference_result else None
    return metric_dict


def CalDice(prediction: torch.Tensor,  # [B, C, H, W, (S)]
            groundTruth: torch.Tensor,  # [B, C, H, W, (S)]
            class_list: List[str],
            metric_name: str = "dice",
            writer: SummaryWriter = None,
            reference_result: bool = False):
    assert prediction.shape == groundTruth.shape
    num_class = len(class_list)
    out_dict = {}
    if reference_result:
        DICE = DiceMetric(include_background=True,
                          get_not_nans=False,
                          ignore_empty=False,
                          num_classes=len(class_list))
        ref_Dices = DICE(prediction, groundTruth).mean(dim=0).tolist()
        if writer is not None:
            write_metric(writer, ref_Dices, "reference_" + metric_name, class_list)
        out_dict["ref_dice"] = ref_Dices
    Dices = []

    for i in range(prediction.shape[0]):
        pred = prediction[i]
        gt = groundTruth[i]
        dices = []

        for target_num in range(num_class):
            pred_mask = pred == target_num
            gt_mask = gt == target_num
            union = (pred_mask | gt_mask).sum()
            intersection = (pred_mask & gt_mask).sum()
            iou = intersection / (1e-8 + union) if union > 0 else torch.tensor(1)
            dice = 2 * iou / (iou + 1)
            dices.append(dice)
        Dices.append(dices)

    Dices = torch.Tensor(Dices).mean(dim=0).tolist()
    if writer is not None:
        write_metric(writer, Dices, metric_name, class_list)
    out_dict["dice"] = Dices
    out_dict["name_string"] = "".join([f"_{name}:{dice:.3}" for name, dice in zip(class_list[1:], Dices[1:])])
    return out_dict


def write_metric(writer, metric, metric_name, class_list):
    print(f"{metric_name}: ")
    metric_dict = {}
    table_html = f"<table><tr><th>Class</th><th>{metric_name}</th></tr>"
    for i, name in enumerate(class_list):
        if name == "BG":
            continue
        table_html += f"<tr><td>{name}</td><td>{metric[i]:.4}</td></tr>"
        print(f"\t {name}: {metric[i]:.4} ")
        metric_dict[name] = metric[i]
    table_html += "</table>"
    writer.add_text(f"metrics/{metric_name}", table_html)
    return metric_dict


def apply_mask(pred, img):
    if pred.shape[1] == 1 and pred.dtype == torch.float32:
        pred = torch.cat([1 - pred, pred], dim=1)
    if pred.dtype == torch.float32:
        pred = torch.argmax(pred, dim=1, keepdim=True)
    pred = torch.eye(pred.max() + 1)[pred.squeeze(1)]
    pred = pred.permute(0, 3, 1, 2)
    if img.shape[1] == 1:
        img = torch.cat([img, img, img], dim=1)
    img = img.permute([0, 2, 3, 1]).to(torch.float32)
    for i in range(1, pred.shape[1]):
        img[pred[:, i] == 1] = COLORS[i - 1]
    img = img.permute([0, 3, 1, 2])
    return torch.clip(img, 0, 1)


def write_val_img(src, out_dict, pred, writer, idx, class_list, entropy=None):
    """
    :param src: [B, 1 or 3, H, W] ranging from [-1, 1]
    :param out_dict: out_dict from dataset
    :param pred: [B, C, H, W], after softmax and detached
    :param writer: tensorboard writer
    :param idx: index of the image
    :param class_list: list of class names
    :param entropy: entropy map
    """
    tb_img = []
    num_class = len(class_list)

    origin = (out_dict['origin'] + 1) / 2
    label = out_dict['label']
    tb_img.append(((src + 1) / 2).to('cpu'))
    tb_img.append(origin.to('cpu'))
    tb_img += [pred[:, i: i + 1, ...] for i in range(num_class)]
    hard_pred = torch.argmax(pred, dim=1, keepdim=True).cpu()
    tb_img.append(apply_mask(hard_pred, origin).to('cpu'))
    tb_img.append(apply_mask(label, origin).to('cpu'))
    tb_img.append(entropy)
    tb_img.append(hard_pred / num_class * 0.8)
    tb_img.append(label.to('cpu') / num_class * 0.8)
    for k in range(len(tb_img)):
        if tb_img[k].shape[1] == 1:
            tb_img[k] = torch.cat([tb_img[k]] * 3, dim=1)
    tb_img = torch.stack(tb_img, dim=2)
    tb_img = rearrange(tb_img, 'b c n h w -> c (b h) (n w)')
    metric_string = CalDice(hard_pred, label, class_list)["name_string"]
    writer.add_image(f'val_res/{idx}{metric_string}', tb_img, dataformats='CHW')
    tb_img = tb_img.permute([1, 2, 0]).detach().cpu().numpy()
    tb_img = (tb_img * 255).astype(np.uint8)
    return tb_img, metric_string


def config_to_html_table(config):
    table_html = "<table>"
    for key, value in vars(config).items():
        table_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
    table_html += "</table>"
    return table_html
