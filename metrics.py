from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics as skm

# =============================================================================
# Logger (inherits configuration from central logger)
# =============================================================================
logger = logging.getLogger(__name__)


# =============================================================================
# Utility meters
# =============================================================================
class AverageMeter:
    """Keeps track of current value, sum, count, and average of a metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class MetricMeter:
    """
    Aggregates true labels and predictions across batches and computes
    padded cmap metrics.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):
        # use round to mitigate label smoothing issues when thresholding in padded_cmap
        self.y_true.extend(y_true.cpu().detach().round().numpy().tolist())
        self.y_pred.extend(y_pred["clipwise_output"].cpu().detach().numpy().tolist())

    @property
    def avg(self):
        y_true_arr = np.array(self.y_true)
        y_pred_arr = np.array(self.y_pred)
        cmap_3 = padded_cmap(y_true_arr, y_pred_arr, padding_factor=3)
        cmap_5 = padded_cmap(y_true_arr, y_pred_arr, padding_factor=5)
        return {
            "cmap_pad_3": cmap_3,
            "cmap_pad_5": cmap_5,
        }


# =============================================================================
# Losses
# =============================================================================
class BCEFocalLoss(nn.Module):
    """
    Binary Cross-Entropy focal loss for addressing class imbalance.
    Reference: adds modulating factor (1-p)^gamma and alpha balancing.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * (1. - probas) ** self.gamma * bce_loss + \
               (1. - targets) * probas ** self.gamma * bce_loss
        return loss.mean()


class BCEFocal2WayLoss(nn.Module):
    """
    Combines primary and auxiliary focal losses: one on input logits and one on
    the max over framewise logits.
    """
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()
        self.focal = BCEFocalLoss()
        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


# =============================================================================
# Criteria wrappers
# =============================================================================
def calc_loss(y_true, y_pred):
    """
    ROC AUC score over flattened predictions and truths.
    """
    return skm.roc_auc_score(np.array(y_true), np.array(y_pred))


def cutmix_criterion(preds, new_targets):
    """
    Loss for CutMix: weighted combination of two targets.
    """
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    criterion = BCEFocal2WayLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def mixup_criterion(preds, new_targets):
    """
    Loss for MixUp: weighted combination of two targets.
    """
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    criterion = BCEFocal2WayLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def loss_fn(logits, targets):
    """
    Primary loss function wrapper.
    """
    loss_fct = BCEFocal2WayLoss()
    return loss_fct(logits, targets)


# =============================================================================
# Metric helpers
# =============================================================================
def padded_cmap(y_true, y_pred, padding_factor=5):
    """
    Compute average precision with padding rows to stabilize against small class counts.
    """
    num_classes = y_true.shape[1]
    pad_rows = np.array([[1] * num_classes] * padding_factor)
    y_true_padded = np.concatenate([y_true, pad_rows])
    y_pred_padded = np.concatenate([y_pred, pad_rows])
    score = skm.average_precision_score(y_true_padded, y_pred_padded, average='macro')
    return score


# =============================================================================
# Self-test
# =============================================================================
if __name__ == '__main__':
    y_true = np.array([[0., 0., 1.], [0., 1., 0.]])
    y_pred = np.array([[0.1, 0.6, 0.75], [0.2, 0.6, 0.3]])
    score = padded_cmap(y_true, y_pred, padding_factor=5)
    print(score)
