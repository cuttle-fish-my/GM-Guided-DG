import copy
import os
from abc import abstractmethod

import blobfile as bf
from einops import rearrange
from torch.nn import functional as F
from torch.optim import AdamW

from GGADG import logger
from GGADG import utils
from GGADG.fp16_util import MixedPrecisionTrainer
from GGADG.losses import *
from GGADG.nn import update_ema
from GGADG.script_util import create_model, args_to_dict, model_defaults
from GGADG.unet import Discriminator

INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            data,
            batch_size,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            writer,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            weight_decay=0.0,
            lr_anneal_steps=5000,
            in_channels=3,
            num_class=4,
    ):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.num_class = num_class
        self.step = 0
        self.resume_step = 0
        self.in_channels = in_channels

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.writer = writer
        if self.resume_step:
            self._load_optimizer_state()

            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                utils.load_state_dict(
                    resume_checkpoint, map_location=utils.dev()
                )
            )

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = utils.load_state_dict(
                ema_checkpoint, map_location=utils.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = utils.load_state_dict(
                opt_checkpoint, map_location=utils.dev()
            )
            self.opt.load_state_dict(state_dict)

    def match_input_channel(self, x):
        if x.shape[1] == 1 and self.in_channels == 3:
            x = th.cat([x, x, x], dim=1)
        elif x.shape[1] == 3 and self.in_channels == 1:
            raise ValueError("Input channel mismatch")
        return x

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    @abstractmethod
    def forward_backward(self, batch, cond):
        pass

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.batch_size)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

    def write_loss(self, losses: dict):
        """
        Write losses to the Tensorboard.
        """
        for k, v in losses.items():
            v = v.mean().item()
            self.writer.add_scalar(f"losses/{k}", v, self.step)

    def write_img(self, img, pred, label=None, others=None):
        """
        Write images to the Tensorboard.
        """
        tb_img = [(img + 1) / 2]
        tb_img += [pred[:, i][:, None] for i in range(1, pred.shape[1])]
        tb_img.append(torch.argmax(pred, dim=1, keepdim=True) / pred.shape[1] * 0.8)
        if label is not None:
            tb_img.append(label / pred.shape[1] * 0.8)
        if others is not None:
            tb_img.append(others)
        for i in range(len(tb_img)):
            if tb_img[i].shape[1] == 1:
                tb_img[i] = tb_img[i].repeat(1, 3, 1, 1)
        tb_img = torch.stack(tb_img, dim=1)
        tb_img = rearrange(tb_img, "b n c h w -> c (b h) (n w)")
        self.writer.add_image(f"{'source' if label is not None else 'target'}/{self.step}", tb_img, dataformats="CHW")


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{step :06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())


class SegmentTrainLoop(TrainLoop):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.CELoss = torch.nn.CrossEntropyLoss()

    def compute_losses(self, batch, label):
        pred = self.model(batch)
        loss = self.CELoss(pred, label[:, 0, ...])
        return {'loss': loss}, {'pred': pred}

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        losses, preds = self.compute_losses(self.match_input_channel(batch), cond['label'])
        loss = losses['loss']
        log_loss_dict({k: v for k, v in losses.items()})
        self.mp_trainer.backward(loss)
        self.write_loss({k: v for k, v in losses.items()})
        if self.step % 100 == 0:
            self.write_img(batch, preds['pred'], cond['label'])


class UDATrainLoop(TrainLoop):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lambda_uda = 1e-3
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.BCELoss = torch.nn.BCEWithLogitsLoss()
        self.UDA_mode = None
        self.discriminator = Discriminator(self.num_class, 64, use_fp16=kwargs['use_fp16']).to(utils.dev())
        self.discriminator_trainer = MixedPrecisionTrainer(
            model=self.discriminator,
            use_fp16=self.use_fp16,
            fp16_scale_growth=kwargs["fp16_scale_growth"],
        )
        self.discriminator_opt = AdamW(
            self.discriminator_trainer.master_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if self.UDA_mode == "advent":
            self.discriminator_trainer.optimize(self.discriminator_opt)
        if took_step:
            self._update_ema()

        self._anneal_lr()
        self.log_step()

    def set_lambda(self, lambda_uda):
        self.lambda_uda = lambda_uda

    def set_UDA_mode(self, mode):
        assert mode in ["minent", "advent"]
        self.UDA_mode = mode

    def compute_losses(self, target, label, source):
        losses = {}
        pred = {}
        target_pred = self.model(target)
        losses["CELoss"] = self.CELoss(target_pred, label[:, 0, ...])

        target_pred = torch.softmax(target_pred, dim=1)
        pred["target_pred"] = target_pred
        pred["target_entropy"] = entropy_map(target_pred, reduce=True)

        source_pred = self.model(source)
        source_pred = torch.softmax(source_pred, dim=1)
        pred["source_pred"] = source_pred

        losses["entropy"], pred["source_entropy"] = entropy_loss(source_pred)
        if self.UDA_mode == "minent":
            losses["loss"] = losses["CELoss"] + self.lambda_uda * losses["entropy"]
        elif self.UDA_mode == "advent":
            target_disc = self.discriminator(entropy_map(target_pred))
            losses["loss_adv"] = self.BCELoss(
                target_disc, torch.zeros_like(target_disc)
            )
            source_disc_detach = self.discriminator(entropy_map(source_pred.detach()))
            target_disc_detach = self.discriminator(entropy_map(target_pred.detach()))
            losses["loss"] = losses["CELoss"] + self.lambda_uda * losses["loss_adv"]
            losses["loss_disc"] = self.BCELoss(
                source_disc_detach, torch.zeros_like(source_disc_detach)
            ) + self.BCELoss(
                target_disc_detach, torch.ones_like(target_disc_detach)
            )
        return losses, pred

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        if self.UDA_mode == "advent":
            self.discriminator_trainer.zero_grad()
        target = self.match_input_channel(batch)
        label = cond["label"]
        source = self.match_input_channel(cond["source"])
        losses, pred = self.compute_losses(target, label, source)
        loss = losses["loss"]
        log_loss_dict({k: v for k, v in losses.items()})
        self.mp_trainer.backward(loss)
        if self.UDA_mode == "advent":
            self.discriminator_trainer.backward(losses["loss_disc"])

        self.write_loss({k: v for k, v in losses.items()})
        if self.step % 100 == 0:
            self.write_img(target, pred["target_pred"], label, others=pred["target_entropy"])
            self.write_img(source, pred["source_pred"], others=pred["source_entropy"])


class TestTimeAdaptationAgent(torch.nn.Module):
    def __init__(self, opts, writer):
        super().__init__()

        model = create_model(Segment=True, num_class=opts.num_class, **args_to_dict(opts, model_defaults().keys()))
        model.load_state_dict(
            utils.load_state_dict(opts.model_path, map_location="cpu")
        )
        model.to(utils.dev())
        self.model = model
        self.config_TTA(opts)
        self.writer = writer

        params, params_name = self.collect_TTA_params()
        self.lr = opts.TTA_lr
        self.optimizer = AdamW(params, lr=self.lr)
        self.steps = opts.TTA_steps
        assert self.steps >= 1, "TTA_steps must be >= 1"
        self.episodic = opts.TTA_episodic
        self.total_step = 0
        self.num_class = opts.num_class
        self.model_state = copy.deepcopy(self.model.state_dict())
        self.optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.lambda_ent = opts.lambda_ent
        self.lambda_consistency = opts.lambda_consistency
        self.class_idx = opts.TTA_class_idx

    def config_TTA(self, opts):
        self.model.train()
        self.model.requires_grad_(False)

        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.GroupNorm):
                m.requires_grad_(True)
                if opts.TTA_mode == "PseudoLabel":
                    m.momentum = opts.lambda_BN
                elif opts.TTA_mode == "Entropy":
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

    def collect_TTA_params(self):
        params = []
        names = []
        for name_m, m in self.model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.GroupNorm):
                for name_p, p in m.named_parameters():
                    if name_p in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{name_m}.{name_p}")
        return params, names

    def forward(self, x, out_dict):
        if self.episodic:
            self.reset()
        preds = []
        entropies = []
        pred = []
        entropy = None

        self.model.eval()

        for step in range(self.steps):
            pred, entropy = self.forward_and_adapt(x)
            preds.append(pred)
            entropies.append(entropy)

        preds = torch.cat([torch.argmax(p, dim=1, keepdim=True) for p in preds], dim=1)
        entropies = torch.cat(entropies, dim=1)
        preds = preds / self.num_class * 0.8
        tb_img = torch.stack([preds, entropies], dim=0)
        tb_img = rearrange(tb_img, "n b c h w -> (n h) (b c w)")
        self.writer.add_image(f"TTA/{int(self.total_step / self.steps) - 1}", tb_img, dataformats="HW")
        return pred, entropy

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        # forward
        pred = self.model(x)
        pred = torch.softmax(pred, dim=1)
        # adapt
        loss, entropy = entropy_loss(pred)
        loss = self.lambda_ent * loss
        hard_pred = torch.argmax(pred, dim=1, keepdim=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_step += 1
        return pred, entropy

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)


class PseudoLabelAgent(TestTimeAdaptationAgent):
    def __init__(self, opts, writer, prior):
        super().__init__(opts, writer)
        assert self.episodic, "PseudoLabelAgent must be episodic"
        self.TTA_alpha = opts.TTA_alpha
        self.global_prior = torch.Tensor(prior)[None, :, None, None]
        self.prior = (1 - self.TTA_alpha) * self.global_prior + self.TTA_alpha / self.num_class

    def forward(self, x, out_dict):
        self.reset()
        preds = []
        entropies = []
        pred = None
        entropy = None
        # linear combination of mean / var between target and source domain
        self.model.train()
        _ = self.model(x)
        self.model.eval()

        self.prior = (1 - self.TTA_alpha) * self.global_prior + self.TTA_alpha / self.num_class

        for step in range(self.steps):
            pred, entropy = self.forward_and_adapt(x)
            preds.append(pred)
            entropies.append(entropy)
        preds = torch.cat([torch.argmax(p, dim=1, keepdim=True) for p in preds], dim=1)
        entropies = torch.cat(entropies, dim=1)
        preds = preds / self.num_class * 0.8
        tb_img = torch.stack([preds, entropies], dim=0)
        tb_img = rearrange(tb_img, "n b c h w -> (n h) (b c w)")
        self.writer.add_image(f"TTA/{int(self.total_step / self.steps) - 1}", tb_img, dataformats="HW")
        pred = pred / (self.prior + 1e-8)
        pred = pred / pred.max()
        return pred, entropy

    def forward_and_adapt(self, x):
        """
        step 1: get posterior from model
        step 2: compute pseudo label = argmax(posterior / prior)
        step 3: compute CE loss and other losses
        step 4: update model
        step 5: update self.prior with pseudo label:
            self.prior = alpha * self.prior + (1 - alpha) * pseudo label prior
        """

        if self.lambda_consistency != 0:
            # Test Time Augmentation
            x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
            pred = self.model(x)
            pred = torch.cat([pred[0:1], torch.flip(pred[1:2], dims=[3])], dim=0)
            # Compute consistency loss
            loss_consistency = consistency_loss(pred)
            pred = pred.mean(dim=0, keepdim=True)
        else:
            pred = self.model(x)
            loss_consistency = 0

        # Get posterior
        posterior = torch.softmax(pred, dim=1)
        # Get pseudo label
        pseudo_label = torch.argmax(posterior / (self.prior + 1e-8), dim=1, keepdim=True)
        # Compute masked CE loss
        mask = pseudo_label == self.class_idx if self.class_idx != -1 else 1
        loss = self.lambda_ent * F.cross_entropy(pred * mask, pseudo_label.squeeze(1))
        loss += self.lambda_consistency * loss_consistency
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update prior
        self.total_step += 1
        new_prior = torch.eye(self.num_class)[pseudo_label.flatten()].sum(dim=0)[None, :, None, None]
        new_prior /= new_prior.sum(dim=1, keepdim=True)
        new_prior = self.TTA_alpha / self.num_class + (1 - self.TTA_alpha) * new_prior
        self.prior = self.TTA_alpha * self.prior + (1 - self.TTA_alpha) * new_prior
        return posterior, entropy_map(posterior, reduce=True)


class TSNEAgent(PseudoLabelAgent):
    def __init__(self, opts, writer, prior):
        super().__init__(opts, writer, prior)

    def forward(self, x, out_dict):
        self.reset()
        preds = []
        features = []
        pred = []
        feature = []

        self.model.train()
        _ = self.model(x)
        self.model.eval()

        self.prior = (1 - self.TTA_alpha) * self.global_prior + self.TTA_alpha / self.num_class

        for step in range(self.steps):
            pred, feature = self.forward_and_adapt(x)
            preds.append(torch.argmax(pred, dim=1).flatten().detach().cpu().numpy())
            features.append(rearrange(feature, "b c h w -> (b h w) c").detach().cpu().numpy())
        return preds, features

    def forward_and_adapt(self, x):

        # Test Time Augmentation
        x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        pred, feature = self.model(x, last_feature=True)
        pred = torch.cat([pred[0:1], torch.flip(pred[1:2], dims=[3])], dim=0)
        feature = torch.cat([feature[0:1], torch.flip(feature[1:2], dims=[3])], dim=0)
        # Compute consistency loss
        loss_consistency = consistency_loss(pred)
        pred = pred.mean(dim=0, keepdim=True)
        feature = feature.mean(dim=0, keepdim=True)

        # Get posterior
        posterior = torch.softmax(pred, dim=1)
        # Get pseudo label
        pseudo_label = torch.argmax(posterior / (self.prior + 1e-8), dim=1, keepdim=True)
        # Compute masked CE loss
        mask = pseudo_label == self.class_idx if self.class_idx != -1 else 1
        loss = self.lambda_ent * F.cross_entropy(pred * mask, pseudo_label.squeeze(1))
        loss += self.lambda_consistency * loss_consistency
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update prior
        self.total_step += 1
        new_prior = torch.eye(self.num_class)[pseudo_label.flatten()].sum(dim=0)[None, :, None, None]
        new_prior /= new_prior.sum(dim=1, keepdim=True)
        new_prior = self.TTA_alpha / self.num_class + (1 - self.TTA_alpha) * new_prior
        self.prior = self.TTA_alpha * self.prior + (1 - self.TTA_alpha) * new_prior
        return posterior, feature
