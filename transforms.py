from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import logging

import numpy as np
import pandas as pd
import torch
import librosa
import albumentations as A
import matplotlib.pyplot as plt
from torchvision import transforms as tr

import colorednoise as cn

from configs import Config, AudioParams

# =============================================================================
# Logger (will inherit level from root; configure via configs.configure_logger)
# =============================================================================
logger = logging.getLogger(__name__)


# =============================================================================
# Audio Transformation Primitives (lightweight analog of albumentations for audio)
# =============================================================================
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply or np.random.rand() < self.p:
            return self.apply(y, sr=sr)
        return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError


class OneOf(Compose):
    """
    Randomly apply one transform from a list, with normalized probabilities.
    """
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps] if s > 0 else [1 / len(transforms)] * len(transforms)

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data


class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        max_vol = np.abs(y).max()
        if max_vol == 0:
            return y
        y_vol = y * (1.0 / max_vol)
        return np.asfortranarray(y_vol)


class NewNormalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        y_mm = y - y.mean()
        max_abs = np.abs(y_mm).max()
        if max_abs == 0:
            return y_mm
        return y_mm / max_abs


class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5):
        super().__init__(always_apply, p)
        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        if a_white == 0:
            return y
        augmented = (y + white_noise * (1.0 / a_white) * a_noise).astype(y.dtype)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        if a_pink == 0:
            return y
        augmented = (y + pink_noise * (1.0 / a_pink) * a_noise).astype(y.dtype)
        return augmented


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_range=5):
        super().__init__(always_apply, p)
        self.max_range = max_range

    def apply(self, y: np.ndarray, sr, **params):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(y, sr, n_steps)
        return augmented


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1):
        super().__init__(always_apply, p)
        self.max_rate = max_rate

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented


# =============================================================================
# Volume utility helpers
# =============================================================================
def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Decrease volume by db decibels.
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Increase volume by db decibels.
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return y * dbs


# =============================================================================
# Alternate colored-noise generator (replicates earlier implementation)
# =============================================================================
def powerlaw_psd_gaussian(exponent, size, fmin=0):
    """Generate Gaussian (1/f)**beta noise (pink/brown etc.) with unit variance.

    Based on Timmer & Koenig (1995). Last axis is time.
    """
    # Ensure size is list for mutability
    try:
        size = list(size)
    except TypeError:
        size = [size]

    samples = size[-1]
    f = np.fft.rfftfreq(samples)

    # Frequency scaling
    s_scale = f
    fmin = max(fmin, 1.0 / samples)
    ix = np.sum(s_scale < fmin)
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.0)

    # Expected standard deviation
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.0
    sigma = 2 * np.sqrt(np.sum(w ** 2)) / samples

    size[-1] = len(f)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(np.newaxis,) * dims_to_add + (Ellipsis,)]

    sr = np.random.normal(scale=s_scale, size=size)
    si = np.random.normal(scale=s_scale, size=size)

    if not (samples % 2):
        si[..., -1] = 0
    si[..., 0] = 0

    s = sr + 1j * si
    y = np.fft.irfft(s, n=samples, axis=-1) / sigma
    return y


# =============================================================================
# Albumentations image transforms (for spectrograms)
# =============================================================================
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

albu_transforms = {
    'train': A.Compose([
        # A.HorizontalFlip(p=0.5),
        # A.OneOf([
        #     A.CoarseDropout(max_holes=8, max_height=5, max_width=16),
        #     A.CoarseDropout(max_holes=4)
        # ], p=0.5),
        A.Normalize(mean, std)
    ]),
    'valid': A.Normalize(mean, std)
}


# =============================================================================
# Feature computation and image conversion
# =============================================================================
def compute_melspec(y, params):
    """
    Compute log-scaled mel-spectrogram from audio signal.
    """
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=params.sr,
        n_mels=params.n_mels,
        fmin=params.fmin,
        fmax=params.fmax
    )
    melspec = librosa.power_to_db(melspec).astype(np.float32)
    return melspec


def get_melspectr(y, params):
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=params.sr,
        n_mels=params.n_mels,
        fmin=params.fmin,
        fmax=params.fmax
    )
    return melspec.astype(np.float32)


def crop_or_pad(y, length, sr, train=True, probs=None):
    """
    Crop or pad a 1D audio array to a fixed length.
    """
    if len(y) <= length:
        y = np.concatenate([y, np.zeros(length - len(y))])
    else:
        if not train:
            start = 0
        elif probs is None:
            start = np.random.randint(len(y) - length)
        else:
            start = (np.random.choice(np.arange(len(probs)), p=probs) + np.random.random())
            start = int(sr * (start))
        y = y[start: start + length]
    return y.astype(np.float32)


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """
    Convert a single-channel spectrogram to 3-channel RGB-like image in [0,255].
    """
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)
    return V


def mono_to_color_v2(X, eps=1e-6):
    """
    Variant of mono_to_color with albumentations normalization applied.
    """
    X = np.stack([X, X, X], axis=-1)
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    normalize = A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    V = (normalize(image=V)['image'] + 1) / 2
    return V


def mono_to_color_v3(X: np.ndarray, len_chack, mean=0.5, std=0.5, eps=1e-6):
    """
    Alternative conversion using torchvision transforms.
    """
    trans = tr.Compose([
        tr.ToPILImage(),
        # tr.Resize([224, 313]),
        tr.ToTensor(),
        tr.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    X = np.stack([X, X, X], axis=-1)
    V = (255 * X).astype(np.uint8)
    V = (trans(V) + 1) / 2
    return V


def random_power(images, power=1.5, c=0.7):
    """
    Raise image intensities to a random power for contrast-like augmentation.
    """
    images = images - images.min()
    images = images / (images.max() + 1e-7)
    images = images ** (random.random() * power + c)
    return images


# =============================================================================
# Mixup / CutMix utilities
# =============================================================================
def rand_bbox(size, lam):
    W = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)

    cx = np.random.randint(W)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)

    return bbx1, bbx2


def cutmix(data, targets, alpha):
    """
    Apply CutMix augmentation on batch.
    Returns modified data and list [targets, shuffled_targets, lambda].
    """
    indices = torch.randperm(data.size(0))
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bbx2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, :] = data[indices, :, bbx1:bbx2, :]

    lam = 1 - (bbx2 - bbx1) / data.size()[-2]
    new_targets = [targets, shuffled_targets, lam]
    return data, new_targets


def mixup(data, targets, alpha):
    """
    Apply MixUp augmentation on batch.
    Returns blended data and list [targets, shuffled_targets, lambda].
    """
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets


# =============================================================================
# Debug / example routines
# =============================================================================
def check_cutmix():
    """
    Sanity check for cutmix by constructing two sample inputs and visualizing results.
    """
    config = Config()
    train = pd.read_csv(config.train_dataframe_path)
    SR = 32000

    sample1 = train.loc[0, :]
    wav_path1 = sample1["filepath"]
    labels1 = sample1["labels"]
    y1 = np.load(wav_path1)[:5 * SR]
    image1 = compute_melspec(y1, AudioParams)
    image1 = mono_to_color(image1).astype(np.uint8).T
    targets1 = np.zeros(len(config.target_columns), dtype=np.float32) + 0.0025
    for ebird_code in labels1.split():
        targets1[config.target_columns.index(ebird_code)] = 0.995

    sample2 = train.loc[200, :]
    wav_path2 = sample2["filepath"]
    labels2 = sample2["labels"]
    y2 = np.load(wav_path2)[:5 * SR]
    image2 = compute_melspec(y2, AudioParams)
    image2 = mono_to_color(image2).astype(np.uint8).T
    targets2 = np.zeros(len(config.target_columns), dtype=np.float32) + 0.0025
    for ebird_code in labels2.split():
        targets2[config.target_columns.index(ebird_code)] = 0.995

    inputs = torch.from_numpy(np.array([image1, image2]))
    targets = torch.from_numpy(np.array([targets1, targets2]))
    logger.info("Input shape: %s", inputs.shape)
    inputs, new_targets = cutmix(inputs, targets, 0.4)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(image1[0, :, :])
    axs[0, 0].set_title('image 1 (before cutmix)')
    axs[0, 1].imshow(image2[0, :, :])
    axs[0, 1].set_title('image 2 (before cutmix)')
    axs[1, 0].imshow(inputs[0][0, :, :])
    axs[1, 0].set_title('image 1 (after cutmix)')
    axs[1, 1].imshow(inputs[1][0, :, :])
    axs[1, 1].set_title('image 2 (after cutmix)')
    plt.show()


# =============================================================================
# Scripted demo when run as main
# =============================================================================
if __name__ == '__main__':
    config = Config()
    # configure_logger should have been called externally if desired
    train = pd.read_csv(config.train_dataframe_path)

    SR = 32000
    sample = train.loc[21, :]
    wav_path = sample["filepath"]
    labels = sample["labels"]
    y = np.load(wav_path)
    logger.info("Audio length (sec): %s", len(y) // SR)

    if len(y) < 5 * SR:
        y = np.concatenate([y, y, y])[:AudioParams.duration * AudioParams.sr]
    y = y[:5 * SR]

    image = compute_melspec(y, AudioParams)
    image = mono_to_color(image).astype(np.uint8)
    logger.info("Spectrogram image shape: %s", image.shape)  # (224, 313, 3)

    image_transformed = albu_transforms['train'](image=image)['image']

    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.imshow(image[:, :, 0])
    plt.subplot(212)
    plt.imshow(image_transformed[:, :, 0])
    plt.show()
