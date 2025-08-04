from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch.optim.lr_scheduler as lr_scheduler
from ast import literal_eval

# =============================================================================
# Logger
# =============================================================================
logger = logging.getLogger(__name__)


# =============================================================================
# Scheduler factory functions
# =============================================================================
def step(optimizer, last_epoch, step_size=80, gamma=0.1, **_):
    """
    Standard StepLR scheduler.
    """
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)


def multi_step(optimizer, last_epoch, milestones=[500, 5000], gamma=0.1, **_):
    """
    MultiStepLR scheduler with specified milestone epochs.
    Milestones can be provided as a string representation (e.g., "[100,200]").
    """
    if isinstance(milestones, str):
        try:
            milestones = literal_eval(milestones)
        except Exception:
            logger.warning("Failed to parse milestones string '%s'; using default.", milestones)
            milestones = [500, 5000]
    return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)


def exponential(optimizer, last_epoch, gamma=0.995, **_):
    """
    Exponential decay scheduler.
    """
    return lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)


def none(optimizer, last_epoch, **_):
    """
    No-op scheduler: effectively keeps LR constant by using a very large step.
    """
    return lr_scheduler.StepLR(optimizer, step_size=10_000_000, last_epoch=last_epoch)


def reduce_lr_on_plateau(optimizer, last_epoch, mode='min', factor=0.1, patience=10,
                         threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, **_):
    """
    ReduceLROnPlateau scheduler (requires calling `.step(metric)` externally).
    """
    logger.info("reduce_lr_on_plateau, factor: %s, patience: %s, last_epoch: %s", factor, patience, last_epoch)
    return lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode=threshold_mode,
        cooldown=cooldown,
        min_lr=min_lr,
    )


def cosine(optimizer, last_epoch, T_max=50, eta_min=1e-5, **_):
    """
    Cosine annealing scheduler.
    """
    logger.info("cosine annealing, T_max: %s, eta_min: %s, last_epoch: %s", T_max, eta_min, last_epoch)
    return lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch
    )


def get_scheduler(config, optimizer, last_epoch):
    """
    Retrieves and instantiates the scheduler specified in config.
    """
    func = globals().get(config.scheduler_name)
    if func is None:
        raise ValueError(f"Scheduler '{config.scheduler_name}' not found.")
    return func(optimizer, last_epoch, **config.scheduler_params)
