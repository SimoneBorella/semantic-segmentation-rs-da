import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_label=255, weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target, balance_weights=[0.4, 1.0], sb_weights=1.0, pixel_wise_weights = None):

        if not (isinstance(score, list) or isinstance(score, tuple)):
            return torch.mean(sb_weights * self._forward(score, target))
        else:
            if len(balance_weights) == len(score):
                if pixel_wise_weights is not None:
                    loss = sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
                    return torch.mean(loss * pixel_wise_weights)
                else:
                    loss = sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
                    return torch.mean(loss)
            elif len(score) == 1:
                return sb_weights * self._forward(score[0], target)
            else:
                raise ValueError("Lengths of prediction and target are not identical!")