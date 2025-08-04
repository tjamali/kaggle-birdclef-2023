from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import logging

import numpy as np
import pandas as pd
import librosa
import torch

from configs import Config, AudioParams
from transforms import (
    Normalize,
    random_power,
    crop_or_pad,
    get_melspectr,
    compute_melspec,
    mono_to_color_v2,
    powerlaw_psd_gaussian,
)

# =============================================================================
# Logger
# =============================================================================
logger = logging.getLogger(__name__)


# Keep the original pattern: global 'config' refers to the class, not an instance.
config = Config


class WaveformDataset(torch.utils.data.Dataset):
    """
    Dataset that builds composite spectrogram images from multiple audio samples.
    Used for BirdCLEF 2023, mixing primary and background bird sounds with augmentation.
    """
    def __init__(self, df: pd.DataFrame, config, mode='train'):
        self.df = df
        self.mode = mode
        self.config = config
        self.stop_border = 0.3  # Probability of stopping mixing early
        self.level_noise = 0.05  # amplitude scaling for added noise
        self.amp_coef = 100  # signal amplification during mixing

        # Base waveform normalization
        self.wave_transforms = Normalize(p=1)
        if self.mode == 'train':
            self.pink_noises = self.generate_pink_noises()

    def generate_pink_noises(self):
        """
        Precompute mel-spectrograms of pink noise for augmentation.
        """
        from tqdm import tqdm  # local import to avoid global dependency if not needed
        pink_noises = []
        for i in tqdm(range(2000), desc="making pink noises for data augmentation"):
            pink_noise = powerlaw_psd_gaussian(1, AudioParams.sr * AudioParams.duration).astype(np.float32)
            pink_noise_image = compute_melspec(pink_noise, AudioParams)
            pink_noises.append(pink_noise_image)
        return pink_noises

    def generate_image(self, index_list):
        """
        Combine multiple audio samples (primary + background) into a single spectrogram image.
        """
        birds, background = [], []
        image = np.zeros((self.config.n_mels, self.config.mel_time_length), dtype=np.float32)

        for i, idy in enumerate(index_list):
            sample = self.df.loc[idy, :]
            audio = np.load(sample["filepath"])

            label = sample["primary_label"]
            if label not in birds:
                birds.append(label)

            # secondary labels as background birds
            secondary_labels = eval(sample.secondary_labels)
            if secondary_labels:
                for bg in secondary_labels:
                    if bg not in background:
                        background.append(bg)

            # Normalize waveform
            audio = self.wave_transforms(audio, sr=AudioParams.sr)

            # Ensure minimum length
            if len(audio) < 5 * AudioParams.sr:
                audio = np.concatenate([audio, audio, audio])[:AudioParams.duration * AudioParams.sr]

            audio = crop_or_pad(
                audio,
                AudioParams.duration * AudioParams.sr,
                sr=AudioParams.sr,
                train=(self.mode == 'train'),
                probs=None
            )

            # Compute mel-spectrogram
            mel = get_melspectr(audio, AudioParams)

            if self.mode == 'train':
                # Contrast adjustment and mixing
                mel = random_power(mel, power=3, c=0.5)
                image += mel * (random.random() * self.amp_coef + 1)
            else:
                image += mel

            # Early stop mixing based on probability
            if random.random() < self.stop_border:
                break

        # Convert to dB and normalize to [0,1]
        image = librosa.power_to_db(image)
        image = (image + 80) / 80

        return image, birds, background

    def transform_image(self, image):
        """
        Apply additional augmentations and convert to 3-channel representation.
        """
        if self.mode == 'train':
            # Add white noise
            if random.random() < 0.9:
                noise = np.random.sample(image.shape).astype(np.float32) + 10
                noise *= image.mean() * self.level_noise * (np.random.sample() + 0.3)
                image += noise

            # Add pink noise (precomputed)
            if random.random() < 0.9:
                noise = random.choice(self.pink_noises)
                noise = (noise - noise.min()) / (noise.max() - noise.min()) + 2.5
                noise *= 4 * image.mean() * self.level_noise * (np.random.sample() + 0.3)
                image += noise

            # Add bandpass-like noise
            if random.random() < 0.9:
                a = random.randint(0, self.config.n_mels // 2)
                b = random.randint(a + 20, self.config.n_mels)
                noise = np.random.sample((b - a, self.config.mel_time_length)).astype(np.float32) + 9
                noise *= 0.05 * image.mean() * self.level_noise * (np.random.sample() + 0.3)
                image[a:b, :] += noise

            # Frequency shaping: reduce upper frequencies
            if random.random() < 0.5:
                image -= image.min()
                r = random.randint(self.config.n_mels // 2, self.config.n_mels)
                x = random.random() / 2
                pink_noise = np.array([
                    np.concatenate((1 - np.arange(r) * x / r, np.zeros(self.config.n_mels - r) - x + 1))
                ]).T
                image *= pink_noise
                image /= image.max()

            # Contrast adjustment
            image = random_power(image, power=2, c=0.7)

        # Convert to 3-channel image and transpose for model input
        image = mono_to_color_v2(image)
        image = image.T  # shape transformation

        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Build sample: select multiple indices for train (mixing), generate image, apply transforms, and produce target vector.
        """
        if self.mode == 'train':
            idx2 = random.randint(0, len(self.df) - 1)
            idx3 = random.randint(0, len(self.df) - 1)
            index_list = [idx, idx2, idx3]
        else:
            index_list = [idx]

        image, birds, background = self.generate_image(index_list)
        image = self.transform_image(image)

        if config.label_smoothing:
            targets = np.zeros(self.config.num_classes, dtype=np.float32) + 0.0025
        else:
            targets = np.zeros(self.config.num_classes, dtype=np.float32)

        # Background birds get lower weight
        if background:
            for bird in background:
                targets[config.target_columns.index(bird)] = 0.3

        # Primary birds
        if config.label_smoothing:
            for bird in birds:
                targets[config.target_columns.index(bird)] = 0.995
        else:
            for bird in birds:
                targets[config.target_columns.index(bird)] = 1.0

        return {"image": image, "targets": targets}


if __name__ == '__main__':
    fold = 0
    train_df = pd.read_csv(config.train_dataframe_path)
    trn_df = train_df[train_df.kfold != fold].reset_index(drop=True)
    train_dataset = WaveformDataset(df=trn_df, config=config, mode='train')
    # trigger a sample retrieval for sanity
    train_dataset.__getitem__(100)
