import argparse
import logging
import pickle
import random
import sys
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

logging.basicConfig(force=True, level=logging.INFO)


def set_seed(seed):
    """Sets random seed everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # use deterministic algorithms
    print("Seed set", seed)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def timestamp():
    return dt.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_config_args(parser, yaml_file_path, dataset):
    """Register every key of the dataset section of a YAML config as an argparse
    argument (so it can be overridden from the command line) and parse."""
    with open(yaml_file_path, "r") as file:
        config = yaml.safe_load(file)[dataset]

    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                parser.add_argument(f"--{key}_{sub_key}", type=type(sub_value), default=sub_value)
        elif isinstance(value, list):
            parser.add_argument(f"--{key}", type=str, nargs="+", default=value)
        elif isinstance(value, bool):
            parser.add_argument(f"--{key}", type=str2bool, default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    return parser.parse_args()


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def plot_and_save_loss(train_losses, val_losses, test_losses, filename):
    """Saves a line plot comparing training/validation/test losses over epochs."""
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.plot(epochs, test_losses, label="Test Loss")

    plt.title("Training vs Validation vs Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(filename, format="png", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved as {filename}")


def get_logger(filepath: str, level=logging.INFO):
    logger = logging.getLogger(__name__)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    fileHandler = logging.FileHandler(filepath)
    streamHandler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        fmt="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger
