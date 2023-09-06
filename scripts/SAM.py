import argparse
import glob
import io
import pathlib
import sys

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms
from einops import rearrange
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from GGADG import utils
from GGADG.image_datasets import ImageDataset
from GGADG.utils import COLORS

COLORS = COLORS.numpy()


def main(opts):
    pathlib.Path(opts.save_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(opts.save_dir)
    dataset = ImageDataset(opts.data_dir, opts.modality, input_mode="origin", mode="val")
    class_list = dataset.meta_info["class_list"]
    num_class = len(class_list)
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Loading model from {opts.model_path}")
    model = sam_model_registry[opts.model_type](checkpoint=opts.model_path)
    model.to(utils.dev())
    predictor = SamPredictor(model)
    labels = {}
    preds = {}
    desc = f"{'pseudo_label' if opts.pseudo_label_dir != '' else 'ground_truth'} prompt"
    for i, (src, out_dict) in enumerate(tqdm(dataLoader, file=sys.stdout, desc=desc)):
        if args.selected_idx != -1 and i != args.selected_idx:
            continue
        file_name = out_dict['file_name']
        patient_idx, slice_idx, num_slice = file_name[0].split('_')
        if labels.get(patient_idx) is None:
            labels[patient_idx] = [torch.empty(0)] * int(num_slice)
            preds[patient_idx] = [torch.empty(0)] * int(num_slice)
        label = out_dict["label"]
        labels[patient_idx][int(slice_idx) - 1] = label

        img = torch.cat([src[0], src[0], src[0]], dim=0).permute([1, 2, 0])
        img = ((img + 1) * 127.5).to(torch.uint8).numpy()

        if opts.pseudo_label_dir != "":
            label = glob.glob(f"{opts.pseudo_label_dir}/{i}_*.png")[0]
            label = torchvision.transforms.ToTensor()(Image.open(label).convert('L')).to(torch.float32)
            label = label * num_class / 0.8
            label = label.round().to(torch.int64)[None, None]

        pred = np.zeros([1, num_class, *img.shape[:2]])
        tb_img = [img.astype(np.float32) / 255]

        predictor.set_image(img)
        for j in range(1, num_class):
            prompt = {}
            prompt.update(generate_bbox_prompt(label == j))
            if class_list[j] == "Myo":
                prompt.update(generate_centroid_prompt(label == j + 1, is_pos=False))
            mask, _, logit = predictor.predict(
                point_coords=prompt.get("coords", None),
                point_labels=prompt.get("labels", None),
                box=prompt.get("bbox", None),
                multimask_output=False,
            )
            if prompt == {}:
                logit = np.ones_like(logit) * (-1e10)
            logit = torchvision.transforms.Resize(img.shape[0], antialias=True)(
                torch.sigmoid(torch.tensor(logit))).numpy()
            pred[:, j, ...] = logit
            tb_img.append(get_logit_img(img, prompt, logit, j) / 255)

        res, pred = get_pred_img(img, pred)
        preds[patient_idx][int(slice_idx) - 1] = torch.Tensor(pred[None])

        tb_img.append(res / 255)
        tb_img.append(pred.transpose([1, 2, 0]).astype(np.float32) / num_class * 0.8)
        tb_img.append(label[0].permute([1, 2, 0]).numpy() / num_class * 0.8)
        for j in range(len(tb_img)):
            if tb_img[j].shape[-1] == 1:
                tb_img[j] = (tb_img[j]).repeat(3, axis=-1)
        tb_img = np.stack(tb_img, axis=0)
        tb_img = rearrange(tb_img, "n h w c -> c h (n w)")
        writer.add_image(f"val/{i}", tb_img, 0, dataformats="CHW")

    labels = labels.values()
    preds = preds.values()

    label_2d = torch.cat([torch.cat(label, dim=0) for label in labels], dim=0)
    pred_2d = torch.cat([torch.cat(pred, dim=0) for pred in preds], dim=0)

    utils.CalDice(pred_2d, label_2d, dataset.meta_info['class_list'], "dice_2d", writer, False)
    if args.selected_idx == -1:
        label_3d = [torch.stack(label, dim=4) for label in labels]
        pred_3d = [torch.stack(pred, dim=4) for pred in preds]
        utils.CalDice3D(pred_3d, label_3d, dataset.meta_info['class_list'], "dice_3d", writer, False)


def get_pred_img(img, pred):
    pred[:, 0, ...] = np.prod(1 - pred[:, 1:, ...], axis=1)
    num_class = pred.shape[1]
    pred /= pred.sum(axis=1, keepdims=True)
    pred = pred.argmax(axis=1)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    for k in range(1, num_class):
        show_mask(pred == k, plt.gca(), color=np.array(COLORS[k - 1]))
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0)
    plt.close()
    buf.seek(0)
    res = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), 1)
    buf.close()
    res = res[..., [2, 1, 0]]  # BGR to RGB
    return cv2.resize(res, img.shape[:2]), pred


def get_logit_img(img, prompt, logit, idx):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    show_box(prompt.get("bbox", None), plt.gca())
    show_points(prompt.get("coords", None), prompt.get("labels", None), plt.gca())
    show_mask(logit[0], plt.gca(), color=np.array(COLORS[idx - 1]))
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0)
    buf.seek(0)
    res = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), 1)
    buf.close()
    res = res[..., [2, 1, 0]]  # BGR to RGB
    plt.close()
    return cv2.resize(res, img.shape[:2])


def show_points(coords, labels, ax, marker_size=375):
    if coords is None or labels is None:
        return
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    if box is None:
        return
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_mask(mask, ax, color=None):
    if color is None:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        if len(color) == 3:
            color = np.concatenate([color, np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape((1, 1, -1))
    ax.imshow(mask_image)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def generate_centroid_prompt(label, is_pos=True):
    label = label[0, 0, ...]
    idx = torch.argwhere(label == 1).to(torch.float32)
    if len(idx) == 0:
        return {}
    x = idx[:, 1].mean()
    y = idx[:, 0].mean()
    centroid = torch.tensor([[x, y]])
    return {
        "coords": centroid.numpy(),
        "labels": np.array([1 if is_pos else 0]),
    }


def generate_bbox_prompt(label):
    label = label[0, 0, ...]
    idx = torch.argwhere(label == 1)
    if len(idx) == 0:
        return {}
    x_min = idx[:, 1].min()
    x_max = idx[:, 1].max()
    y_min = idx[:, 0].min()
    y_max = idx[:, 0].max()
    bbox = torch.tensor([x_min, y_min, x_max, y_max])
    return {
        "bbox": bbox.numpy(),
    }


def generate_grid_prompt(label, pos_grid_interval=20, neg_grid_interval=20):
    pos_grid = torch.zeros(label.shape[-2:])
    pos_grid[int(pos_grid_interval / 2)::pos_grid_interval, int(pos_grid_interval / 2)::pos_grid_interval] = 1
    neg_grid = torch.zeros(label.shape[-2:])
    neg_grid[int(neg_grid_interval / 2)::neg_grid_interval, int(neg_grid_interval / 2)::neg_grid_interval] = 1
    foreground = pos_grid * label[0, 0, ...]
    background = neg_grid * ~label[0, 0, ...]
    foreground_coords = torch.argwhere(foreground)[:, [1, 0]]
    background_coords = torch.argwhere(background)[:, [1, 0]]
    coords = torch.cat([foreground_coords, background_coords], dim=0)
    labels = torch.cat([torch.ones(foreground_coords.shape[0]), torch.zeros(background_coords.shape[0])], dim=0)
    return {
        "coords": coords.numpy(),
        "labels": labels.numpy(),
    }


def generate_random_prompt(label, N_pos=5, N_neg=5):
    label = label[0, 0, ...]
    foreground = label == 1
    background = label == 0
    foreground_coords = torch.argwhere(foreground)
    foreground_coords = foreground_coords[torch.randperm(foreground_coords.shape[0])[:N_pos]]
    foreground_coords = foreground_coords[:, [1, 0]]
    background_coords = torch.argwhere(background)
    background_coords = background_coords[torch.randperm(background_coords.shape[0])[:N_neg]]
    background_coords = background_coords[:, [1, 0]]
    coords = torch.cat([foreground_coords, background_coords], dim=0)
    labels = torch.cat([torch.ones(foreground_coords.shape[0]), torch.zeros(background_coords.shape[0])], dim=0)
    return {
        "coords": coords.numpy(),
        "labels": labels.numpy(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/MS-CMRSeg2019/val")
    parser.add_argument("--modality", type=str, default="target")
    parser.add_argument("--model_path", type=str, default="../sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="vit_h")
    parser.add_argument("--save_dir", type=str, default="SAM_res/pseudo_label")
    parser.add_argument("--pseudo_label_dir", type=str, default="")
    parser.add_argument("--selected_idx", type=int, default=-1, help="NEVER TOUCH THIS! ONLY FOR DEBUGGING!")
    args = parser.parse_args()
    main(args)
