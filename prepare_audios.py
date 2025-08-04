import os
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa as lb
from joblib import Parallel, delayed
from tqdm import tqdm

# =============================================================================
# Settings / constants
# =============================================================================
"""
This script reads .ogg audio files, trims/resamples them to a fixed length
(defined by USE_SEC), and saves them as numpy arrays for downstream usage in
the BirdCLEF 2023 pipeline.
"""

ROOT = Path('/home/tj/PycharmProjects/kaggle/BirdCLEF_2023')
AUDIO_PATH = ROOT / 'data' / 'train_audio'
AUDIO_PATH_ADDITIONAL = ROOT / 'additional_audios'
METADATA_PATH = ROOT / 'data' / 'train_metadata.csv'

SR = 32000
USE_SEC = 30  # seconds to keep per audio (truncate longer recordings)
NUM_WORKERS = -2  # joblib uses all but one core by default when negative

# Output directory prefix for saved numpy files
OUTPUT_BASE = ROOT / 'train_np'

# =============================================================================
# Logging
# =============================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# =============================================================================
# Utility functions
# =============================================================================
def audio_to_array(audiopath: str) -> np.ndarray:
    """
    Load audio from file, convert to mono, resample if needed, and truncate to USE_SEC.

    Args:
        audiopath: Path to the .ogg audio file.

    Returns:
        1D numpy array of audio samples at SR sampling rate and length <= USE_SEC*SR.
    """
    y, sr = sf.read(audiopath, dtype="float32", always_2d=True)
    y = np.mean(y, axis=1)  # collapse stereo to mono

    if sr != SR:
        y = lb.resample(y, orig_sr=sr, target_sr=SR, res_type='kaiser_fast')

    if len(y) > SR * USE_SEC:
        y = y[:SR * USE_SEC]
    return y


def save_audio(audiopath: str):
    """
    Converts and saves a single audio file to numpy format preserving directory structure.

    Args:
        audiopath: full path to input audio file
    """
    try:
        rel_path = Path(audiopath).relative_to(AUDIO_PATH)
    except Exception:
        try:
            rel_path = Path(audiopath).relative_to(AUDIO_PATH_ADDITIONAL)
        except Exception:
            # fallback: use last two components
            rel_path = Path(*Path(audiopath).parts[-2:])

    save_path = OUTPUT_BASE / rel_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    arr = audio_to_array(audiopath)
    np.save(save_path, arr)
    logger.debug("Saved %s -> %s", audiopath, save_path)


# =============================================================================
# Main processing
# =============================================================================
def build_metadata():
    """
    Construct the merged training DataFrame with primary_label and filename.
    """
    paths_1 = sorted(glob.glob(str(AUDIO_PATH / '*/*.ogg')))
    paths_2 = sorted(glob.glob(str(AUDIO_PATH_ADDITIONAL / '*/*.ogg')))
    paths = paths_1 + paths_2

    labels = [Path(p).parent.name for p in paths]
    filenames = [str(Path(p).parent / Path(p).name) for p in paths]  # preserve the two-level structure
    df_audio = pd.DataFrame({'primary_label': labels, 'filename': filenames})

    train = pd.read_csv(METADATA_PATH)
    merged = pd.merge(train, df_audio, how='outer', on='filename')
    merged['primary_label_x'] = merged['primary_label_y']
    merged.drop(columns='primary_label_y', inplace=True)
    merged.rename(columns={'primary_label_x': 'primary_label'}, inplace=True)
    merged = merged.sort_values(by=['primary_label', 'filename']).reset_index(drop=True)
    return merged, paths


def main():
    # Prepare metadata and list of audio files
    train_df, paths = build_metadata()
    classes = train_df.primary_label.unique()

    # Ensure output directories exist per class
    for cls in tqdm(classes, desc="creating class directories"):
        (OUTPUT_BASE / cls).mkdir(parents=True, exist_ok=True)

    # Process and save audio files in parallel
    logger.info("Processing %d audio files with %s workers", len(paths), NUM_WORKERS)
    Parallel(n_jobs=NUM_WORKERS)(
        delayed(save_audio)(audiopath) for audiopath in tqdm(paths, desc="saving audio arrays")
    )


if __name__ == '__main__':
    main()
