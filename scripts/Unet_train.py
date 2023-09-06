"""
Train a diffusion model on images.
"""

import argparse

from torch.utils.tensorboard import SummaryWriter

from guided_diffusion import utils
from guided_diffusion import logger
from guided_diffusion.image_datasets import ImageDataset, UnpairedDataset, load_data
from guided_diffusion.script_util import (
    model_defaults,
    create_model,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import SegmentTrainLoop, UDATrainLoop


def main():
    args = create_argparser().parse_args()
    logger.configure(dir=args.save_dir)
    logger.log("creating model ...")
    dataset = getDataset(args)
    data = load_data(
        dataset=dataset,
        batch_size=args.batch_size,
    )
    # add num_class to args
    args.num_class = len(dataset.meta_info["class_list"])
    model = create_model(Segment=True, num_class=args.num_class,
                         **args_to_dict(args, model_defaults().keys()))
    model.to(utils.dev())
    logger.log("creating data loader...")

    logger.log("training...")
    TrainLoop = getTrainLoop(args, model, data)
    TrainLoop.run_loop()


def getDataset(args):
    dataset = UnpairedDataset if args.use_UDA else ImageDataset
    return dataset(
        image_paths=args.data_dir,
        modality=args.modality,
        input_mode=args.input_mode,
        heavy_aug=args.heavy_aug,
    )


def getTrainLoop(args, model, data):
    writer = SummaryWriter(args.save_dir)
    config_string = utils.config_to_html_table(args)
    writer.add_text("config", config_string)
    TrainLoop = UDATrainLoop if args.use_UDA else SegmentTrainLoop
    TrainLoop = TrainLoop(
        model=model,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        writer=writer,
        num_class=args.num_class,
    )
    if args.use_UDA:
        TrainLoop.set_lambda(args.lambda_UDA)
        TrainLoop.set_UDA_mode(args.UDA_mode)
    return TrainLoop


def create_argparser():
    defaults = dict(
        data_dir="",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=10000,
        batch_size=1,
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1,
        save_interval=200,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # new added:
        save_dir="saved_models",
        modality="target",
        input_mode="magnitude",
        use_UDA=False,
        UDA_mode="advent",
        lambda_UDA=1e-3,
        heavy_aug=False,
    )
    defaults.update(model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
