"""EmotionHeart training entry point.

Trains the EmotionHeart model (heterogeneous conversation graph + Graphormer
encoder + cross-modal graph contrastive learning) on IEMOCAP or MELD.

Usage (from the repository root):
    python main.py --dataset iemocap --specific True
    python main.py --dataset meld --specific True
"""

import argparse
import os

import torch

import graphdata as gdt
import models
import utils
from data import get_MELD_loaders


def build_graph_datasets(args):
    """Load cached graph datasets, building them from the feature pickle if needed."""
    data_dir = os.path.join(os.getcwd(), args.data_dir_path, args.dataset)
    args.data = os.path.join(data_dir, "data_" + args.dataset + ".pkl")

    graph_trainset_file = os.path.join(data_dir, "graph_trainset.pkl")
    graph_testset_file = os.path.join(data_dir, "graph_testset.pkl")

    # NOTE: the caches bake in the data split and the graph hyperparameters
    # (relation_type, max_dist, ...). Delete data/<dataset>/graph_*.pkl after
    # changing either, so they are rebuilt (preprocess/iemocap_data_split.py
    # does this automatically).
    cache_missing = not (
        os.path.exists(graph_trainset_file) and os.path.exists(graph_testset_file)
    )

    if args.dataset == "iemocap":
        if cache_missing:
            data = utils.load_pkl(args.data)
            if not os.path.exists(graph_trainset_file):
                utils.save_pkl(gdt.iemocap_4_graphDataset(data["train"], "train", args), graph_trainset_file)
            if not os.path.exists(graph_testset_file):
                utils.save_pkl(gdt.iemocap_4_graphDataset(data["test"], "test", args), graph_testset_file)

    elif args.dataset == "meld":
        if cache_missing:
            train_loader, _, test_loader, _ = get_MELD_loaders(args.batch_size, args.data)
            if not os.path.exists(graph_trainset_file):
                utils.save_pkl(gdt.meld_graphDataset(train_loader, "train", args), graph_trainset_file)
            if not os.path.exists(graph_testset_file):
                utils.save_pkl(gdt.meld_graphDataset(test_loader, "test", args), graph_testset_file)

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    trainset = utils.load_pkl(graph_trainset_file)
    testset = utils.load_pkl(graph_testset_file)

    args.n_max_utterances = trainset.n_max_utterances
    args.n_max_speakers = trainset.n_max_speakers

    return trainset, testset


def main(args):
    utils.set_seed(args.seed)

    for directory in ("log", args.save_model_checkpoint, args.save_analysis_path):
        os.makedirs(directory, exist_ok=True)
    log = utils.get_logger("./log/train.log")

    log.info("Load finetuning dataset... Name: " + args.dataset)
    trainset, testset = build_graph_datasets(args)

    log.debug("Building EmotionHeart model...")
    if args.unimodal_inference and args.modalities in ["a", "t", "v"]:
        # Missing-modality experiment: reuse the multimodal (atv) checkpoint and
        # fine-tune only the fusion layer / classifier (and, in the modality-
        # agnostic setting, the shared encoder) for unimodal inference.
        model = torch.load(
            os.path.join(args.save_model_checkpoint, "atv_best_model.pt"),
            weights_only=False,
            map_location=args.device,
        ).to(args.device)

        trainable_prefixes = ("linear_fusion", "classifier")
        if not args.specific:
            trainable_prefixes += ("encoder.",)
        for name, param in model.named_parameters():
            if not name.startswith(trainable_prefixes):
                param.requires_grad = False

        model.args = args
        model.modalities = args.modalities
        model.n_modalities = len(args.modalities)
        model.encoder.args = args
        model.encoder.n_modalities = len(args.modalities)

        total_params = sum(p.numel() for p in model.parameters())
        log.info(f"Total parameters: {total_params:,}")
    else:
        if args.dataset == "iemocap":
            n_nodes = trainset.n_max_utterances
        else:
            n_nodes = max(trainset.n_max_utterances, testset.n_max_utterances)
        encoder = models.EmotionHeartEncoder(args, n_nodes)
        model = models.EmotionHeartModel(args, encoder).to(args.device)

    opt = models.Optim(
        float(args.learning_rate),
        int(args.T),
        float(args.max_grad_value),
        float(args.weight_decay),
        int(args.epochs),
        int(args.n_train_dialogues // args.batch_size),
    )
    opt.set_parameters(model.parameters(), args.optimizer)
    sched = opt.get_scheduler(args.scheduler)

    coach = models.Coach(trainset, testset, testset, model, opt, sched, args, log)

    log.info("Start training...")
    ret = coach.train()

    metrics = {
        "best_dev_f1": ret[0],
        "best_dev_acc": ret[1],
        "best_epoch": ret[2],
        "best_state": ret[3],
        "train_losses": ret[4],
        "dev_losses": ret[5],
        "dev_f1s": ret[6],
        "test_f1s": ret[7],
        "dev_accs": ret[8],
        "test_accs": ret[9],
        "test_losses": ret[10],
    }
    timestamp = utils.timestamp()
    utils.plot_and_save_loss(
        ret[4], ret[5], ret[10],
        filename=os.path.join(os.getcwd(), args.save_analysis_path, f"loss_plot_{timestamp}.png"),
    )
    torch.save(metrics, os.path.join(os.getcwd(), args.save_analysis_path, f"metrics_{timestamp}.pt"))


def get_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="EmotionHeart training")

    parser.add_argument(
        "--specific",
        type=utils.str2bool,
        default=True,
        help="modality-specific encoders (True) or a single modality-agnostic encoder (False).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="iemocap",
        choices=["iemocap", "meld"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--relation_type",
        default="eam",
        choices=["e", "ea", "eam"],
        help="Graph relation types. e: inter-speaker, a: intra-speaker, m: inter-modality.",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["sgd", "rmsprop", "adam", "adamw"],
        help="Optimizer.",
    )

    temp, _ = parser.parse_known_args()

    setting = "specific" if temp.specific else "agnostic"
    config_file = os.path.join("config", f"{temp.dataset}_{setting}.yaml")
    args = utils.get_config_args(parser, config_file, dataset=temp.dataset)

    args.num_edges = len(args.relation_type)
    args.num_degree = len(args.relation_type)
    if args.relation_type == "eam" and args.modalities == "atv":
        args.num_degree += 1

    args.ffn_embed_dim = args.encoder_embed_dim * args.ffn_embed_scaler

    args.dataset_embedding_dims = {
        "iemocap": {"a": 100, "t": 768, "v": 512},
        "meld": {"a": 300, "t": 600, "v": 342},
    }

    return args


if __name__ == "__main__":
    main(get_args())
