import os
import time
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm

from configs import Config, configure_logger
from transforms import colored
from metrics import AverageMeter, MetricMeter, loss_fn
from dataset import WaveformDataset
from model import SED
from checkpoint import get_initial_checkpoint, load_checkpoint, save_checkpoint
from scheduler import get_scheduler

# ------------------------------------------------------------------
# Experiment setup
# ------------------------------------------------------------------
config = Config  # keep original pattern (class-level access)
config.exp_id = 47
config.debug = False

# configure logging verbosity based on config.debug
configure_logger(config)
logger = torch.loggers if False else None  # placeholder to signal logging is configured; internal modules use logging.getLogger

device = config.device
set_seed = None  # safety placeholder; will override below if available
# import set_seed from wherever it's defined (configs)
from configs import set_seed  # explicit
set_seed(config.seed)

# rare birds that should be kept across folds
rare_birds = ['brcwea1', 'lotcor1', 'whhsaw1']


# ------------------------------------------------------------------
# Training / validation functions
# ------------------------------------------------------------------
def train_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=config.apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader), leave=True)

    current_lr = optimizer.param_groups[0]['lr']

    for data in tk0:
        optimizer.zero_grad()
        inputs = data['image'].to(device)
        targets = data['targets'].to(device)

        with torch.cuda.amp.autocast(enabled=config.apex):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg, lr=current_lr)
    return scores.avg, losses.avg


def valid_fn(model, data_loader, device):
    model.eval()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader), leave=False)

    with torch.no_grad():
        for data in tk0:
            inputs = data['image'].to(device)
            targets = data['targets'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


# ------------------------------------------------------------------
# Main run loop
# ------------------------------------------------------------------
def run(train_df):
    if config.debug:
        train_df = train_df.head(200)

    gc.collect()

    train_score_dict = {}
    valid_score_dict = {}
    train_loss_dict = {}
    valid_loss_dict = {}

    for fold in range(5):
        if fold not in config.folds:
            continue

        config.train_dir = config.root / f"models/Exp_{config.exp_id}" / f"fold_{fold}"
        os.makedirs(config.train_dir, exist_ok=True)

        print("=" * 100)
        print(f"Fold {fold} Training")
        print("=" * 100)

        # ensure rare birds get moved out of validation
        for bird_name in rare_birds:
            train_df.loc[train_df.primary_label == bird_name, 'kfold'] = (fold + 1) % 5

        trn_df = train_df[train_df.kfold != fold].reset_index(drop=True)
        val_df = train_df[train_df.kfold == fold].reset_index(drop=True)

        train_dataset = WaveformDataset(df=trn_df, config=config, mode='train')
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train_bs,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=True
        )

        valid_dataset = WaveformDataset(df=val_df, config=config, mode='valid')
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.valid_bs,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False
        )

        model = SED(
            base_model_name=config.base_model_name,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            in_channels=config.in_channels
        )
        model = model.to(device)

        warnings.filterwarnings("ignore")

        optimizer = transformers.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        checkpoint = get_initial_checkpoint(config)
        if checkpoint is not None:
            last_epoch, _ = load_checkpoint(model, optimizer, checkpoint)
        else:
            last_epoch, _ = -1, -1

        print(f'from checkpoint: {checkpoint} -- last epoch:{last_epoch}')
        scheduler = get_scheduler(config, optimizer, last_epoch)

        best_val_loss = np.inf
        train_scores = []
        valid_scores = []
        train_losses = []
        valid_losses = []
        start_epoch = last_epoch + 1

        for epoch in range(start_epoch, config.epochs):
            start_time = time.time()

            train_score, train_loss = train_fn(model, train_dataloader, device, optimizer, scheduler)
            valid_score, valid_loss = valid_fn(model, valid_dataloader, device)

            # Update scheduler (ReduceLROnPlateau uses metric)
            if config.scheduler_name == 'reduce_lr_on_plateau':
                scheduler.step(valid_loss)
            else:
                scheduler.step()

            train_scores.append(train_score)
            valid_scores.append(valid_score)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            elapsed = time.time() - start_time

            result = (
                f"[{epoch:02d}] loss: ({train_loss:.5f}, {valid_loss:.5f}) "
                f"-- cmap 5: ({train_score['cmap_pad_5']:0.5f}, {valid_score['cmap_pad_5']:0.5f}) "
                f"-- time: {elapsed:.0f}s"
            )
            if valid_loss < best_val_loss:
                print(colored([100, 255, 100], result))
                best_val_loss = valid_loss
                name = os.path.join(
                    config.train_dir,
                    f"Exp_{config.exp_id}_fold_{fold}_epoch_{epoch:02d}_loss_{valid_loss:0.5f}_cmap_{valid_score['cmap_pad_5']:0.5f}"
                )
                save_checkpoint(config, model, optimizer, epoch, name=name)
            else:
                print(result)

        train_score_dict[f'{fold}'] = train_scores
        valid_score_dict[f'{fold}'] = valid_scores
        train_loss_dict[f'{fold}'] = train_losses
        valid_loss_dict[f'{fold}'] = valid_losses

        # cleanup
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

    # save run metrics
    np.save(config.train_dir / 'train_score_dict.npy', train_score_dict)
    np.save(config.train_dir / 'valid_score_dict.npy', valid_score_dict)
    np.save(config.train_dir / 'train_loss_dict.npy', train_loss_dict)
    np.save(config.train_dir / 'valid_loss_dict.npy', valid_loss_dict)


if __name__ == '__main__':
    train_df = pd.read_csv(config.train_dataframe_path)
    print('number of bird classes:', train_df.primary_label.nunique())
    run(train_df)
