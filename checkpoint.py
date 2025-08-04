from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import logging

import torch

logger = logging.getLogger(__name__)


def get_last_checkpoint(checkpoint_dir):
    """
    Return the most recent checkpoint file in a directory matching pattern 'Exp_*.pth'.
    If none exist, returns None.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = [
        ckpt for ckpt in os.listdir(checkpoint_dir)
        if ckpt.startswith('Exp_') and ckpt.endswith('.pth')
    ]
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints)
    return os.path.join(checkpoint_dir, checkpoints[-1])


def get_initial_checkpoint(config):
    """
    Wrapper to get the last checkpoint from config.train_dir.
    """
    checkpoint_dir = config.train_dir
    return get_last_checkpoint(checkpoint_dir)


def get_checkpoint(config, name):
    """
    Construct full path for a checkpoint with given name inside config.train_dir.
    """
    checkpoint_dir = config.train_dir
    return os.path.join(checkpoint_dir, name)


def copy_last_n_checkpoints(config, n, name):
    """
    Copy the last `n` epoch_* checkpoints (sorted lexicographically) to new names.
    New names are formatted via `name.format(i)` where i is 0-based index among the last n.
    """
    checkpoint_dir = os.path.join(config.train_dir, 'checkpoint')
    if not os.path.isdir(checkpoint_dir):
        logger.warning("Checkpoint directory does not exist: %s", checkpoint_dir)
        return

    checkpoints = [
        ckpt for ckpt in os.listdir(checkpoint_dir)
        if ckpt.startswith('epoch_') and ckpt.endswith('.pth')
    ]
    if not checkpoints:
        return

    checkpoints = sorted(checkpoints)
    last_n = checkpoints[-n:]
    for i, checkpoint in enumerate(last_n):
        src = os.path.join(checkpoint_dir, checkpoint)
        dst = os.path.join(checkpoint_dir, name.format(i))
        shutil.copyfile(src, dst)
        logger.debug("Copied checkpoint %s to %s", src, dst)


def load_checkpoint(model, optimizer, checkpoint):
    """
    Load a checkpoint file into model and optimizer.

    Returns:
        last_epoch (int), step (int)
    """
    logger.info("Loading checkpoint from %s", checkpoint)
    checkpoint_data = torch.load(checkpoint)

    # Rebuild state_dict, stripping 'module.' if present and skipping batch-tracked buffers
    checkpoint_dict = {}
    for k, v in checkpoint_data.get('state_dict', {}).items():
        if 'num_batches_tracked' in k:
            continue
        if k.startswith('module.'):
            checkpoint_dict[k[7:]] = v
        else:
            checkpoint_dict[k] = v

    model.load_state_dict(checkpoint_dict)

    if optimizer is not None and 'optimizer_dict' in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data['optimizer_dict'])

    step = checkpoint_data.get('step', -1)
    last_epoch = checkpoint_data.get('epoch', -1)

    logger.debug("Loaded checkpoint: epoch=%s, step=%s", last_epoch, step)
    return last_epoch, step


def save_checkpoint(config, model, optimizer, epoch, step=0, weights_dict=None, name=None):
    """
    Save model and optimizer state to a checkpoint file.

    If `name` is provided, uses that as filename; otherwise uses epoch_{:04d}.pth.
    """
    checkpoint_dir = config.train_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    if name:
        checkpoint_path = os.path.join(checkpoint_dir, f'{name}.pth')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch:04d}.pth')

    if weights_dict is None:
        weights_dict = {
            'state_dict': model.state_dict(),
            'optimizer_dict': optimizer.state_dict() if optimizer is not None else None,
            'epoch': epoch,
            'step': step,
        }
    torch.save(weights_dict, checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)
