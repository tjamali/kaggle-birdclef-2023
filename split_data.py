"""
Build stratified k-fold splits for BirdCLEF 2023 and merge audio filepaths.

- Merges existing metadata with preprocessed .npy audio files.
- Creates combined labels (primary + secondary) and computes label lengths.
- Assigns folds via StratifiedKFold on primary_label.
- Writes updated train.csv back to disk.
"""

import ast
import glob
import logging

import pandas as pd
from path import Path
from sklearn.model_selection import StratifiedKFold

# ----------------------
# Configuration
# ----------------------
SEED = 8
N_FOLDS = 5
ROOT = Path('/home/tj/PycharmProjects/kaggle/BirdCLEF_2023')
TRAIN_CSV_PATH = ROOT / 'train.csv'
NPY_GLOB_PATTERN = ROOT / 'train_np' / '*' / '*.npy'

# ----------------------
# Logger
# ----------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def build_dataframe():
    # gather all npy paths
    all_paths = sorted(glob.glob(str(NPY_GLOB_PATTERN)))
    logger.info("Found %d .npy audio files", len(all_paths))

    # load metadata
    train = pd.read_csv(TRAIN_CSV_PATH)
    train['secondary_labels'] = train['secondary_labels'].fillna('[]')

    # build combined labels string
    def combine_labels(primary, secondary_str):
        try:
            secondary = ast.literal_eval(secondary_str)
        except Exception:
            secondary = []
        return " ".join([primary] + secondary)

    train['labels'] = train.apply(lambda row: combine_labels(row['primary_label'], row['secondary_labels']), axis=1)
    train['len_labels'] = train['labels'].map(lambda x: len(x.split()))

    # build path dataframe matching format used elsewhere: "class/filename" without extension
    path_df = pd.DataFrame({'filepath': all_paths})
    def filename_key(p):
        p_obj = Path(p)
        parent = p_obj.parent.name
        name = p_obj.name
        name_no_ext = name[:-4] if name.lower().endswith('.npy') else name
        return f"{parent}/{name_no_ext}"
    path_df['filename'] = path_df['filepath'].map(filename_key)

    # merge
    merged = pd.merge(train, path_df, on='filename', how='inner')

    return merged


def assign_folds(df):
    # deal with rare birds info (for logging/inspection)
    birds_freq = df.primary_label.value_counts(ascending=True)
    rare = birds_freq[birds_freq < 5]
    if not rare.empty:
        logger.info("Birds with fewer than 5 examples (should be present in all folds):\n%s", rare)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df['kfold'] = -1
    for n, (_, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        df.loc[val_idx, 'kfold'] = n
    df['kfold'] = df['kfold'].astype(int)
    return df


def main():
    df = build_dataframe()
    df = assign_folds(df)
    # overwrite train.csv in ROOT
    df.to_csv(TRAIN_CSV_PATH, index=False)
    logger.info("Wrote updated train.csv with kfolds to %s", TRAIN_CSV_PATH)


if __name__ == '__main__':
    main()
