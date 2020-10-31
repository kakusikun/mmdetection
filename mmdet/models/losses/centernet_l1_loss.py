import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

@LOSSES.register_module()
class CenterNetRegL1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(CenterNetRegL1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                mask,
                ind):

        pred = _transpose_and_gather_feat(pred, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

