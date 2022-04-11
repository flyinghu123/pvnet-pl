import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..optimizer.lr_scheduler import MultiStepLR
from typing import Dict
from omegaconf import DictConfig
from src.models.pvnet import PvnetModelResnet18
import torch.nn.functional as F


def softmax_focal_loss(inputs,
                       targets,
                       alpha: float = 0.25,
                       gamma: float = 2,
                       reduction: str = 'mean'):
    r"""
    :param inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
    :param targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
    :param alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
    :param gamma: Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples.
    :param reduction: 'none' | 'mean' | 'sum'
                        'none': No reduction will be applied to the output.
                        'mean': The output will be averaged.
                        'sum': The output will be summed.
    :return: Loss tensor with the reduction option applied.

    .. math::
        FL = -\alpha (1-p)^\gamma \log(p)
    .. math::
        p = \frac{e^i}{\sum_j e^j}

    >>> targets = torch.ones((1, 1, 2, 2)).to(torch.int64)
    >>> inputs = torch.cat([torch.zeros_like(targets), torch.ones_like(targets)], dim=1).float()
    >>> softmax_focal_loss(inputs, targets, reduction='mean')
    tensor(0.0057)
    >>> targets = torch.zeros((1, 1, 2, 2)).to(torch.int64)
    >>> inputs = torch.cat([torch.ones_like(targets), torch.zeros_like(targets)], dim=1).float()
    >>> softmax_focal_loss(inputs, targets, reduction='mean')
    tensor(0.0170)
    >>> targets = torch.tensor([1, 0, 0, 1]).view(1, 1, 2, 2)
    >>> inputs = torch.cat([1 - targets, targets], dim=1).float()
    >>> softmax_focal_loss(inputs, targets, reduction='mean')
    tensor(0.0113)
    """
    if targets.dim() != inputs.dim():
        targets = targets.unsqueeze(1)

    p = torch.softmax(inputs, dim=1).detach()
    log_softmax = F.log_softmax(inputs, dim=1)
    p = p.gather(1, targets.to(torch.int64))
    log_softmax = log_softmax.gather(1, targets.to(torch.int64))

    p_t = p

    loss = -log_softmax * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class LitPvnet(pl.LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # self.model = get_wheat_model(self.cfg)
        self.model = PvnetModelResnet18((1 + 8) * 2, 2)
        self.vote_crit = nn.SmoothL1Loss(reduction='sum')
        self.seg_crit = nn.CrossEntropyLoss()
        # if cfg.pvnet.seg_loss == 'focal':
        #     self.seg_crit = softmax_focal_loss
        # if cfg.pvnet.use_proxy_loss:
        #     if cfg.pvnet.proxy_loss == 'proxy':
        #         self.proxy_crit = ProxyVotingLoss()
        #     elif cfg.pvnet.proxy_loss == 'proxy_v2':
        #         self.proxy_crit = ProxyVotingLossV2()
        #     else:
        #         raise ValueError(f'{cfg.pvnet.seg_loss} is not exist!')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        output = self(batch['inp'])
        weight = batch['mask'][:, None]  # .float()
        sample_num = max(weight.sum(), 1)
        # keypoint loss
        vertex_target, kpt_2d = batch['vertex'], batch['kpt_2d']
        vertex_pred = output['vertex']
        sc, c = vertex_target.shape[1], vertex_pred.shape[1]
        bn = vertex_pred.shape[0]
        total_ver_loss = 0
        loss = 0
        # if not (cfg.pvnet.use_expectation_loss and batch['epoch'] >= cfg.pvnet.use_expectation_epoch):
        for j in range(bn):
            ver_loss = None
            # ver_loss_stats = None
            for i in range(sc // c):  # symmetric object
                beg, end = (i * c), (i * c + c)
                cur_loss = self.compute_vertex_loss(
                    vertex_pred[j, None], vertex_target[j, None, beg:end],
                    kpt_2d[j, None,
                           beg // 2:end // 2], sample_num, weight[j, None])
                # if ver_loss is None or cur_ver_loss_stats['proxy_loss'] < ver_loss_stats['proxy_loss']:
                # if ver_loss is None or cur_loss < ver_loss:
                ver_loss = cur_loss
                total_ver_loss += ver_loss
                # ver_loss_stats = cur_ver_loss_stats
        loss += total_ver_loss
        mask = batch['mask'].long()
        seg_loss = 1.0 * self.seg_crit(output['seg'],
                                       mask)  # cfg.pvnet.seg_weight: 1.0
        loss += seg_loss
        self.log('train_ver_loss', total_ver_loss, prog_bar=True)
        self.log('train_seg_loss', seg_loss, prog_bar=True)
        self.log('loss', loss, prog_bar=True)
        return loss
        # return 0

    def compute_vertex_loss(self, ver_pred, ver_target, kpt_2d_gt, sample_num,
                            weight):
        loss = 0
        b, c, h, w = ver_pred.shape
        # proxy loss
        # if
        # vote loss
        vote_loss = self.vote_crit(ver_pred * weight, ver_target * weight)
        vote_loss = 1.0 * vote_loss / sample_num / c  # cfg.pvnet.vote_weight: 1.0
        loss += vote_loss
        return loss



    def configure_optimizers(self):
        lr = self.cfg.optimizers.lr
        weight_decay = self.cfg.optimizers.weight_decay
        params = []
        for key, value in self.model.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(self.model.parameters(),
        #                              lr=lr,
        #                              weight_decay=weight_decay)
        scheduler = MultiStepLR(optimizer,
                                milestones=self.cfg.train.milestones,
                                gamma=self.cfg.train.gamma)
        return [optimizer], [scheduler]
