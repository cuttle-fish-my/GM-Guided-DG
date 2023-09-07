import math

import torch
import torch as th


def entropy_loss(pred):
    b, c, h, w = pred.shape
    entropy = entropy_map(pred, reduce=True)
    loss = th.sum(entropy) / (b * h * w)
    return loss, entropy


def entropy_map(pred, reduce=False):
    entropy = - pred * th.log(pred + 1e-30) / math.log(pred.shape[1])
    # The value range of entropy is [0, log(c)], so we normalize it to [0, 1]
    if reduce:
        return th.sum(entropy, dim=1, keepdim=True)
    else:
        return entropy


def consistency_loss(pred):
    b, c, h, w = pred.shape
    loss = torch.nn.L1Loss()(pred[0], pred[1]) / 2
    return loss
