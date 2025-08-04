# BirdCLEF 2023 Audio Classification / Sound Event Detection

[![Kaggle BirdCLEF 2023](https://img.shields.io/badge/Kaggle-BirdCLEF_2023-blue?logo=kaggle)](https://www.kaggle.com/competitions/birdclef-2023) [![Status](https://img.shields.io/badge/status-experimental-yellow)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

## TL;DR

This repository contains a reproducible pipeline for the **BirdCLEF 2023** Kaggle competition: Train a sound event detection model to identify bird species from long environmental audio recordings (BirdCLEF 2023). Audio is converted into mel-spectrograms, augmented/mixed (CutMix/MixUp, noise, contrast), and fed into attentive deep models; evaluation uses padded class-wise average precision (cmap) to handle imbalance. 

### Dataset expectations / minimal setup

To run training end-to-end you need the official BirdCLEF 2023 data arranged as follows:

1. **Raw audio files** (`*.ogg`): organized by primary label, e.g.:  
   `data/train_audio/<primary_label>/*.ogg` and any extras under `additional_audios/`. These are preprocessed into fixed-length NumPy arrays via `prepare_audios.py` and saved to `train_np/<primary_label>/<filename>.npy`.   
2. **Metadata**: the provided `train_metadata.csv` is merged with the generated `.npy` paths in `split_data.py`, secondary labels are cleaned, combined label strings computed, and stratified k-fold splits assigned (based on `primary_label`), producing `train.csv` with a `kfold` column.   
3. **Workflow**: run `prepare_audios.py` → `split_data.py` → `train.py` to start training. The model builds composite spectrograms with primary/background mixing and trains with label smoothing and adaptive scheduling. 

## 📦 Repository Overview / File Graph

prepare\_audios.py           # ingest and preprocess raw .ogg -> numpy arrays
split\_data.py              # merge metadata, assign stratified k-folds
transforms.py             # audio/image augmentations & utilities (spectrograms, CutMix/MixUp, coloring)
dataset.py                # dataset that builds composite spectrograms + target vectors
model.py                  # model architectures (TimmSED, SED, attention blocks, helpers)
metrics.py                # loss definitions, metric aggregation (padded cmap), utility meters
scheduler.py              # learning rate schedulers factory
checkpoint.py             # checkpoint saving/loading with resume support
train.py                 # main training loop tying everything together
plot\_loss\_metric.py       # visualization of loss / metrics over epochs
configs.py               # centralized experiment configuration & logger setup
imports.py               # shared lower-level imports & version helper (used transitively)

### High-level Flow

1. **Audio Preparation**: `prepare_audios.py` converts raw `.ogg` audio into trimmed/resampled `.npy` arrays.
2. **Data Splitting**: `split_data.py` merges audio metadata with available `.npy` files, builds label strings, and assigns stratified k-folds on `primary_label`.
3. **Transforms**: `transforms.py` defines augmentations (audio-level and image-level), mel-spectrogram computation, CutMix/MixUp utilities, and conversion to 3-channel images.
4. **Dataset**: `dataset.py` creates mixed spectrogram images per sample (mixing primary/background birds), applies augmentations, and constructs target vectors with label smoothing.
5. **Model**: `model.py` implements the backbone architectures (EfficientNet via `timm`), attention blocks, and SED composition.
6. **Training**: `train.py` orchestrates training/validation loops, handles rare-bird logic, checkpointing, learning rate scheduling, and logging.
7. **Metrics**: `metrics.py` defines focal losses, aggregation meters, and the padded cmap scoring used for evaluation.
8. **Checkpointing**: `checkpoint.py` handles saving/loading to resume runs safely.
9. **Scheduler**: `scheduler.py` provides flexible LR scheduler selection via config.
10. **Visualization**: `plot_loss_metric.py` plots training/validation loss and cmap score per fold.

## ⚙️ Setup & Dependencies

This project relies on specific versions of its core Python libraries to ensure reproducibility. The required packages (with pinned versions) exist in requirements.txt:

```text
numpy==1.21.6
pandas==1.3.5
scikit-learn==1.0.2
torch==1.13.0
timm==0.6.13
transformers==4.27.4
librosa==0.10.0.post2
soundfile==0.11.0
torchlibrosa==0.1.0
albumentations==1.3.0
matplotlib==3.5.3
joblib==1.2.0
tqdm==4.64.1
colorednoise==2.1.0
````

Install packages with:

```bash
pip install -r requirements.txt
```


## 🧪 Configuration

All experiment-level settings live in `configs.py`. Key fields include:

* `exp_id`, `seed`, `epochs`, `folds`, `lr`, `weight_decay`
* `train_bs`, `valid_bs`, `base_model_name` (e.g., `"tf_efficientnet_b0_ns"`)
* `scheduler_name` and `scheduler_params` (e.g., `'reduce_lr_on_plateau'`)
* `label_smoothing`, `pretrained`, audio/spec parameters (`sample_rate`, `period`, `n_mels`, etc.)
* `debug`: set to `True` to run with a small subset and increased verbosity.

Logger configuration is centralized: after instantiating `Config`, call `configure_logger(config)` so that `config.debug` controls verbosity.

Example:

```python
from configs import Config, configure_logger

config = Config()
configure_logger(config)
```

## 🚀 Training

1. **Prepare audio arrays** (one-time):

```bash
python prepare_audios.py
```

2. **Build train split with k-folds**:

```bash
python split_data.py
```

This will update `train.csv` with a `kfold` column.

3. **Run training**:

```bash
python train.py
```

This script:

* Loads and splits the data.
* Handles rare bird classes to ensure they stay in training folds.
* Builds datasets with augmentation.
* Instantiates model (`SED`), optimizer, scheduler.
* Resumes from checkpoints if available.
* Saves best checkpoints based on validation loss.
* Stores training histories (loss/score) per fold.

### Notes

* Scheduler stepping: `reduce_lr_on_plateau` expects `scheduler.step(metric)` while others use `scheduler.step()`.
* Checkpoints are saved with descriptive filenames including epoch, loss, and cmap score.
* Rare birds (`brcwea1`, `lotcor1`, `whhsaw1`) are forcefully adjusted in folds to avoid them being absent from training splits.

## 🧠 Data Augmentation

* **Audio-level**: normalization, noise injection (white, pink), pitch shift, time stretch.
* **Spectrogram-level**: random power contrast changes, CutMix, MixUp, frequency shaping.
* **Label smoothing**: controlled via config; softens targets (e.g., primary bird gets 0.995 instead of 1.0).

## 📌 Checkpointing & Resuming

`checkpoint.py` exposes helpers to load the latest saved model or specific checkpoint. The training loop calls:

```python
checkpoint = get_initial_checkpoint(config)
if checkpoint:
    last_epoch, _ = load_checkpoint(model, optimizer, checkpoint)
```

Ensure `config.train_dir` is properly set before training (the script does this per fold).

## 📊 Evaluation / Visualization

After or during training, visualize metrics with:

```bash
python plot_loss_metric.py <train_dir> [fold] [optional_save_path]
```

Example:

```bash
python plot_loss_metric.py models/Exp_47/fold_0 0 fold0.png
```

This plots:

* Training/validation loss.
* `cmap_pad_5` score (padded class-wise average precision).

## 🛠 Example Minimal Run

```python
from configs import Config, configure_logger
from train import run
import pandas as pd

config = Config()
configure_logger(config)
train_df = pd.read_csv(config.train_dataframe_path)
run(train_df)
```

## 📁 Suggested Repo Structure

```
.
├── configs.py
├── imports.py
├── prepare_audios.py
├── split_data.py
├── transforms.py
├── dataset.py
├── model.py
├── metrics.py
├── scheduler.py
├── checkpoint.py
├── train.py
├── plot_loss_metric.py
├── train.csv
├── train_np/           # output audio arrays
├── models/             # saved checkpoints + history
└── README.md
```

## License
MIT License

Copyright (c) 2023 Tayeb Jamali

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

