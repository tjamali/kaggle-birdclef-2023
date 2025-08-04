from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging

import numpy as np
import pandas as pd
import torch
from pathlib import Path

# =============================================================================
# Module Overview
# =============================================================================
"""
This module defines core configuration and utility routines for the BirdCLEF 2023
Kaggle competition. The task is to identify bird species from audio recordings
(soundscape / audio clips) using deep learning on spectrogram representations.

Key design elements reflected here:
  * Mel-spectrogram based input (audio -> image-like representation).
  * EfficientNet backbone (via `base_model_name`) as feature extractor.
  * AUC-based evaluation, typical in imbalanced multi-class / multi-label setups.
  * Use of label smoothing, learning rate scheduling, and optional mixed precision (apex).
  * Reproducibility via seed-setting and deterministic PyTorch/cuDNN configuration.
  * Configurable logging verbosity via the `debug` flag.
"""
# =============================================================================


class Config:
    """
    Experiment configuration container for BirdCLEF 2023 model training / evaluation.

    Many fields are class-level for convenient global access. Note that `target_columns`
    and `num_classes` are resolved at import time by reading the training CSV.
    """
    ######################
    # Experiment / optimization
    ######################
    exp_id = 1
    seed = 8
    epochs = 120
    folds = [1]  # subset of folds to run; full CV would be [0,1,2,3,4]
    n_folds = 5
    lr = 1e-3
    weight_decay = 1e-6
    train_bs = 32  # training batch size
    valid_bs = 32  # validation batch size
    base_model_name = "tf_efficientnet_b0_ns"  # backbone architecture from timm

    early_stopping = True
    scheduler_name = 'reduce_lr_on_plateau'  # alternative: 'cosine'
    scheduler_params = {'factor': 0.9, 'patience': 5}
    num_workers = 0  # DataLoader workers
    debug = False  # enable debug mode (controls logging verbosity)
    evaluation = 'AUC'  # primary metric for leaderboard / early stopping
    apex = True  # use mixed precision if available

    ######################
    # Paths & environment
    ######################
    root = Path('/home/tj/PycharmProjects/kaggle/BirdCLEF_2023')
    train_dataframe_path = Path('train.csv')
    pretrained_checkpoint = "/home/tj/PycharmProjects/kaggle/BirdCLEF_2023/pretraining/models/Exp_4/Exp_4_fold_0_epoch_25_loss_0.00240_cmap_0.76698.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ######################
    # Model / label configuration
    ######################
    label_smoothing = True  # smooth labels to mitigate overconfidence
    pooling = "max"  # feature pooling strategy before classification head
    pretrained = True  # use ImageNet / pretraining weights for backbone
    in_channels = 3  # input channels, since mel-spectrogram treated as RGB-like

    # Derive target labels from the train CSV (bird species / primary labels)
    target_columns = pd.read_csv(train_dataframe_path).primary_label.unique().tolist()
    num_classes = len(target_columns)

    ######################
    # Input representation / audio
    ######################
    img_size = 224  # output image size for model (mel bins x time trimmed/resized)

    # Spectrogram specifics tuned for bird audio (typical frequency ranges)
    period = 5  # seconds of audio considered per sample
    mel_time_length = 313  # temporal dimension of mel-spectrogram output
    n_mels = img_size  # number of mel bins (height of spectrogram)
    fmin = 20  # lower frequency for mel
    fmax = 16000  # upper frequency (birds often vocalize well within this)
    n_fft = 2048  # FFT window size
    hop_length = 512  # hop between frames
    sample_rate = 32000  # resample audio to this for consistency / efficiency


class AudioParams:
    """
    Convenience container for audio preprocessing parameters derived from Config.
    Used when constructing mel-spectrograms or other audio-based features.
    """
    sr = Config.sample_rate
    duration = Config.period
    n_mels = Config.n_mels
    fmin = Config.fmin
    fmax = Config.fmax


def set_seed(seed=42):
    """
    Enforce deterministic behavior across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Seed value to apply.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic settings: may slow training but ensures repeatability
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logger(config):
    """
    Configure the root logger based on the config.debug flag.
    If debug is True -> DEBUG level, else INFO.
    """
    level = logging.DEBUG if getattr(config, "debug", False) else logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Avoid duplicate handlers if already configured
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(level)
    root_logger.debug("Logger configured. Debug mode: %s", getattr(config, "debug", False))


def colored(rgb, text):
    """
    Apply 24-bit RGB ANSI coloring to terminal text. Useful for highlighted console logging.

    Args:
        rgb: Three-element sequence of R,G,B in [0,255].
        text: Text to colorize.

    Returns:
        ANSI escaped string with applied color.
    """
    if len(rgb) != 3:
        raise ValueError("rgb must be a sequence of three integers (R,G,B)")
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m{text} \033[38;2;255;255;255m"


if __name__ == '__main__':
    # Example usage / sanity check: replicates original behavior.
    config = Config()
    configure_logger(config)
    # Will raise ValueError if 'abethr1' is not among primary_label values.
    print(config.target_columns.index('abethr1'))
