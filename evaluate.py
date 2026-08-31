"""EmotionHeart checkpoint evaluation.

Loads a trained checkpoint and reports the test-set weighted F1 / accuracy and
the per-class classification report (representative results of the paper).

Usage (from the repository root):
    python evaluate.py --dataset iemocap --specific True
    python evaluate.py --dataset meld --specific True
    python evaluate.py --dataset meld --specific True --checkpoint path/to/model.pt
"""

import argparse
import copy
import os

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

import utils
from main import build_graph_datasets, get_args

DATASET_LABELS = {
    "iemocap": ["hap", "sad", "neu", "ang", "exc", "fru"],
    "meld": ["neu", "sur", "fea", "sad", "joy", "dis", "ang"],
}


def load_testset(args):
    data_dir = os.path.join(os.getcwd(), args.data_dir_path, args.dataset)
    graph_testset_file = os.path.join(data_dir, "graph_testset.pkl")
    if os.path.exists(graph_testset_file):
        return utils.load_pkl(graph_testset_file)
    _, testset = build_graph_datasets(args)
    return testset


@torch.no_grad()
def evaluate(model, testset, device):
    model.eval()
    golds, preds = [], []
    for idx in tqdm(range(len(testset)), desc="evaluate"):
        # deepcopy so the dataset's stored batch dict keeps its CPU tensors
        data = copy.deepcopy(testset[idx])
        for k, v in data.items():
            if v is not None:
                data[k] = v.to(device)
        _, logits, labels, _, _, _ = model(data, testset.n_max_utterances)
        golds.append(labels.detach().cpu())
        preds.append(logits.detach().cpu())
    golds = torch.cat(golds, dim=-1).numpy()
    preds = np.argmax(torch.cat(preds, dim=-1).numpy(), axis=1)
    return golds, preds


def main():
    parser = argparse.ArgumentParser(description="EmotionHeart evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to model_checkpoints/<experiment>/<modalities>_best_model.pt",
    )
    args = get_args(parser)

    checkpoint = args.checkpoint or os.path.join(
        args.save_model_checkpoint, f"{args.modalities}_best_model.pt"
    )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    testset = load_testset(args)

    print(f"Load checkpoint: {checkpoint}")
    model = torch.load(checkpoint, weights_only=False, map_location=args.device)
    model.args.device = args.device

    golds, preds = evaluate(model, testset, args.device)

    w_f1 = metrics.f1_score(golds, preds, average="weighted")
    acc = metrics.accuracy_score(golds, preds)

    print(
        metrics.classification_report(
            golds, preds, target_names=DATASET_LABELS[args.dataset],
            digits=4, zero_division=0,
        )
    )
    print(f"[{args.dataset}] checkpoint: {checkpoint}")
    print(f"Test weighted F1: {w_f1 * 100:.2f} | Test accuracy: {acc * 100:.2f}")


if __name__ == "__main__":
    main()
