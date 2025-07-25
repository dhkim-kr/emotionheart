from data import get_MELD_loaders
import argparse

import os
import torch

import models

import utils
import graphdata as gdt

from datetime import datetime as dt


log = utils.get_logger('./log/train.log')

def main(args):
    utils.set_seed(args.seed)

    # load data
    log.info("Load finetuning dataset... Name: " + args.dataset)
    finetuning_data_dir = os.path.join(os.getcwd(), args.data_dir_path, args.dataset)
    args.data = os.path.join(finetuning_data_dir, "data_" + args.dataset + ".pkl")


    if args.dataset == "iemocap_4" or args.dataset == "iemocap":
        finetuning_data = utils.load_pkl(args.data)
        graph_trainset_file = os.path.join(finetuning_data_dir, "graph_trainset.pkl")
        graph_devset_file = os.path.join(finetuning_data_dir, "graph_devset.pkl")
        graph_testset_file = os.path.join(finetuning_data_dir, "graph_testset.pkl")

        if not os.path.exists(graph_trainset_file):
            trainset = gdt.iemocap_4_graphDataset(finetuning_data["train"], 'train', args)
            utils.save_pkl(trainset, graph_trainset_file)
        trainset = utils.load_pkl(graph_trainset_file)

        args.n_max_utterances = trainset.n_max_utterances
        args.n_max_speakers = trainset.n_max_speakers

        if not os.path.exists(graph_devset_file):
            devset = gdt.iemocap_4_graphDataset(finetuning_data["dev"], 'dev', args)
            utils.save_pkl(devset, graph_devset_file)
        devset = utils.load_pkl(graph_devset_file)

        if not os.path.exists(graph_testset_file):
            testset = gdt.iemocap_4_graphDataset(finetuning_data["test"], 'test', args)
            utils.save_pkl(testset, graph_testset_file)
        testset = utils.load_pkl(graph_testset_file)

    elif args.dataset == "meld":
        pass
    elif args.dataset == "mosei":
        pass

    log.debug("Building graphormer...")

    encoder = models.EmotionHeartEncoder(args)
    model = models.EmotionHeartModel(args, encoder).to(args.device)

    opt = models.Optim(float(args.learning_rate), int(args.T), float(args.max_grad_value), float(args.weight_decay), int(args.epochs), int(args.n_train_dialogues // args.batch_size))
    opt.set_parameters(model.parameters(), args.optimizer)
    sched = opt.get_scheduler(args.scheduler)

    coach = models.Coach_Crossdata(trainset, devset, testset, model, opt, sched, args, log)

    # Train and eval
    log.info("Start training...")
    ret = coach.train()
    # Save.
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
    save_loss_plot_path = os.path.join(os.getcwd(), args.save_analysis_path+'_'+args.dataset,
                                       "loss_plot_"+ dt.now().strftime('%Y-%m-%d-%H-%M-%S')+".png")
    save_metrics_plot_path = os.path.join(os.getcwd(), args.save_analysis_path+'_'+args.dataset,
                                       "metrics_"+ dt.now().strftime('%Y-%m-%d-%H-%M-%S')+".png")
    utils.plot_and_save_loss(ret[4], ret[5], ret[10], filename=save_loss_plot_path)
    torch.save(metrics, save_metrics_plot_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="pretraining_data")

    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="iemocap",
        choices=["iemocap", "iemocap_4", "mosei", "meld"],
        help="Dataset name."
    )
    parser.add_argument(
        "--relation_type",
        default="eam",
        choices=["e", "ea", "eam"],
        help="Choose relation contruct type. e: interlocuter, a: intralocuter, m: intermodality",
    )

    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["sgd","rmsprop","adam","adamw"],
        help="Choose optimizer",
    )

    temp = parser.parse_args()

    args = utils.get_config_args(parser, 'config/'+temp.dataset+'.yaml', dataset=temp.dataset)

    args.num_edges = len(args.relation_type)
    args.num_degree = len(args.relation_type)
    if args.relation_type == "eam" and args.modalities == "atv":
        args.num_degree += 1

    args.ffn_embed_dim = args.encoder_embed_dim * args.ffn_embed_scaler

    args.dataset_embedding_dims = {
        "iemocap": {
            "a": 100,
            "t": 768,
            "v": 512,
        },
        "iemocap_4": {
            "a": 100,
            "t": 768,
            "v": 512,
        },
        "mosei": {
            "a": 80,
            "t": 768,
            "v": 35,
        },
        "meld": {
            "a":300,
            "t":600,
            "v":342
        }
    }

    main(args)
