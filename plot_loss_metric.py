import argparse
import numpy as np
import matplotlib.pyplot as plt
from path import Path


def load_score_dicts(train_dir):
    """
    Load saved metric and loss dictionaries from a training directory.
    """
    def load(path):
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")
        return np.load(str(path), allow_pickle=True).item()

    train_score_dict = load(train_dir / Path("train_score_dict.npy"))
    valid_score_dict = load(train_dir / Path("valid_score_dict.npy"))
    train_loss_dict = load(train_dir / Path("train_loss_dict.npy"))
    valid_loss_dict = load(train_dir / Path("valid_loss_dict.npy"))

    return train_score_dict, valid_score_dict, train_loss_dict, valid_loss_dict


def plot_fold(train_score_dict, valid_score_dict, train_loss_dict, valid_loss_dict, fold, save_path=None):
    """
    Plot loss and cmap_pad_5 score for a given fold.
    """
    fold_key = str(fold)
    if fold_key not in train_loss_dict:
        raise KeyError(f"Fold {fold} not found in train_loss_dict keys: {list(train_loss_dict.keys())}")

    train_loss = train_loss_dict[fold_key]
    valid_loss = valid_loss_dict[fold_key]
    train_score = [x["cmap_pad_5"] for x in train_score_dict[fold_key]]
    valid_score = [x["cmap_pad_5"] for x in valid_score_dict[fold_key]]

    n_epochs = len(train_loss)
    epochs = np.arange(n_epochs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax1.plot(epochs, train_loss, "-o", label="train loss")
    ax1.plot(epochs, valid_loss, "-o", label="valid loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(f"Fold {fold} Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(epochs, train_score, "-o", label="train cmap_pad_5")
    ax2.plot(epochs, valid_score, "-o", label="valid cmap_pad_5")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("cmap score")
    ax2.set_title(f"Fold {fold} cmap_pad_5")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training/validation loss and cmap scores for a given fold.")
    parser.add_argument("--train_dir", required=True, help="Directory where train/valid score/loss .npy files are saved.")
    parser.add_argument("--fold", type=int, default=0, help="Fold number to plot.")
    parser.add_argument("--save", default=None, help="Optional path to save the figure (e.g., out.png).")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    train_score_dict, valid_score_dict, train_loss_dict, valid_loss_dict = load_score_dicts(train_dir)
    plot_fold(train_score_dict, valid_score_dict, train_loss_dict, valid_loss_dict, fold=args.fold, save_path=args.save)


if __name__ == "__main__":
    main()
