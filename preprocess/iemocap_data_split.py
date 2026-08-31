"""Rebuild the IEMOCAP train/dev/test split used in the paper.

Merges the train/dev/test dialogues stored in ``data/iemocap/data_iemocap.pkl``
into a single 151-dialogue pool and re-partitions it with the fixed dialogue
indices below (108 train / 12 dev / 31 test). The file is overwritten in place,
and any stale ``graph_*.pkl`` caches in the same directory are removed so they
are rebuilt from the new split on the next run.

Notes on the released split (kept exactly as used for the paper):
- The 12 dev indices are a subset of the train indices; the dev set is not
  used for model selection in the released code (``models/Coach.py`` selects
  on the test set it receives as ``devset``), so this has no effect on the
  reported results.
- 12 of the 151 dialogues belong to no split and are excluded.
- Train and test indices are disjoint.

The script refuses to run twice: once the file already contains the paper
split, re-running would re-partition an already-partitioned pool and corrupt
the data.

Usage (from the repository root):
    python preprocess/iemocap_data_split.py --data_dir_path data --seed 42
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils

TRAIN_DIALOGUE_IDX = [
    82, 14, 35, 1, 69, 13, 6, 16, 105, 45, 0, 83, 102, 67, 85, 90, 51, 104,
    112, 47, 24, 42, 81, 100, 89, 122, 41, 123, 93, 128, 33, 63, 113, 34, 39,
    101, 110, 31, 18, 17, 136, 142, 80, 15, 73, 68, 19, 148, 27, 86, 56, 141,
    146, 87, 62, 2, 98, 28, 59, 133, 129, 50, 97, 135, 143, 96, 21, 137, 140,
    132, 10, 126, 70, 7, 55, 79, 116, 130, 94, 92, 75, 49, 8, 40, 149, 134,
    32, 84, 12, 60, 11, 139, 118, 77, 131, 109, 29, 72, 52, 30, 145, 91, 3,
    106, 5, 120, 37, 107,
]
DEV_DIALOGUE_IDX = [42, 51, 56, 100, 60, 93, 37, 80, 50, 29, 84, 47]
TEST_DIALOGUE_IDX = [
    144, 78, 150, 53, 22, 108, 124, 74, 4, 95, 138, 20, 115, 38, 46, 57, 25,
    111, 58, 23, 44, 117, 64, 147, 88, 26, 119, 121, 125, 9, 43,
]


def label_distribution(dialogues, num_classes):
    labels = [label for d in dialogues for label in d["labels"]]
    counts = np.bincount(np.array(labels), minlength=num_classes)
    return counts, np.std(counts)


def already_split(dataset):
    """True if the file already contains the paper split (running again would
    corrupt it: the pool would be a re-partitioned pool, not the original)."""
    if (len(dataset["train"]), len(dataset["dev"]), len(dataset["test"])) != (
        len(TRAIN_DIALOGUE_IDX), len(DEV_DIALOGUE_IDX), len(TEST_DIALOGUE_IDX)
    ):
        return False
    train_label_seqs = {tuple(d["labels"]) for d in dataset["train"]}
    return all(tuple(d["labels"]) in train_label_seqs for d in dataset["dev"])


def main(args):
    utils.set_seed(args.seed)

    data_dir = os.path.join(args.data_dir_path, args.dataset)
    data_file = os.path.join(data_dir, "data_" + args.dataset + ".pkl")
    print("Load dataset:", data_file)
    dataset = utils.load_pkl(data_file)

    if already_split(dataset):
        print("data file already contains the paper split; nothing to do.")
        return

    pool = dataset["train"] + dataset["dev"] + dataset["test"]
    n_pool = len(pool)
    max_idx = max(TRAIN_DIALOGUE_IDX + DEV_DIALOGUE_IDX + TEST_DIALOGUE_IDX)
    if n_pool <= max_idx:
        raise SystemExit(
            f"Expected a pool of at least {max_idx + 1} dialogues, got {n_pool}. "
            "The data file does not look like the original IEMOCAP feature pickle; aborting."
        )

    new_dataset = {
        "train": [pool[i] for i in TRAIN_DIALOGUE_IDX],
        "dev": [pool[i] for i in DEV_DIALOGUE_IDX],
        "test": [pool[i] for i in TEST_DIALOGUE_IDX],
    }

    utils.save_pkl(new_dataset, data_file)

    # The graph caches bake in the split; drop them so they are rebuilt.
    for cache in glob.glob(os.path.join(data_dir, "graph_*.pkl")):
        os.remove(cache)
        print("Removed stale graph cache:", cache)

    n_total = len(TRAIN_DIALOGUE_IDX) + len(DEV_DIALOGUE_IDX) + len(TEST_DIALOGUE_IDX)
    print(f"Total number of dialogues: {n_total}")
    for split in ("train", "dev", "test"):
        counts, std = label_distribution(new_dataset[split], args.num_classes)
        print(f"{split.capitalize()} sample count: {counts} (std {std:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="iemocap", choices=["iemocap"])
    parser.add_argument("--data_dir_path", type=str, default="data")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
