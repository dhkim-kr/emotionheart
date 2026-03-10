
import argparse

import utils
from sklearn.manifold import TSNE
import numpy as np
import os
import umap

def main(args):
    utils.set_seed(args.seed)

    type = 1

    if type == 0:
        data_file_name = "train_atv_fused_emb.npy"
        label_file_name = "train_atv_golds_preds.npy"

        X = np.load(os.path.join(args.save_analysis_path, data_file_name))
        Y = np.load(os.path.join(args.save_analysis_path, label_file_name))[0]

        print(f"Acc: {np.sum(Y==np.load(os.path.join(args.save_analysis_path, label_file_name))[1])/3012*100}")

        print("\nRunning t-SNE...")

        dr_type = 1

        if dr_type == 0:
            reducer = umap.UMAP(n_neighbors=20, min_dist=0.2, n_components=2, random_state=args.seed)
            X = reducer.fit_transform(X)
            print(f"Data shape after UMAP: {X.shape}")

        elif dr_type == 1:
            tsne = TSNE(n_components=2, perplexity=50, max_iter=5000, random_state=args.seed)
            X = tsne.fit_transform(X)
            print(f"Data shape after t-SNE: {X.shape}")

        else:
            return

        if args.dataset =="iemocap":
            utils.discrimination_save_scatter(X,Y,args.save_analysis_path,1)
        else:
            utils.discrimination_meld_save_scatter(X, Y, args.save_analysis_path, 1)

    if type == 1:
        # data_file_name = "train_atv_graphs.npy"
        data_file_name = "train_atv_init_emb.npy"
        label_file_name = "train_atv_golds_preds.npy"

        X = np.load(os.path.join(args.save_analysis_path, data_file_name)).reshape(-1,384)
        Y = np.load(os.path.join(args.save_analysis_path, label_file_name))[0]

        print(f"Acc: {np.sum(Y == np.load(os.path.join(args.save_analysis_path, label_file_name))[1]) / 3012 * 100}")

        print("\nRunning t-SNE...")

        dr_type = 1

        if dr_type == 0:
            reducer = umap.UMAP(n_neighbors=20, min_dist=0.2, n_components=2, random_state=args.seed)
            X = reducer.fit_transform(X)
            print(f"Data shape after UMAP: {X.shape}")

        elif dr_type == 1:
            tsne = TSNE(n_components=2, perplexity=50, max_iter=500, random_state=args.seed)
            X = tsne.fit_transform(X)
            print(f"Data shape after t-SNE: {X.shape}")

        else:
            return

        X = X.reshape(3,-1,2)
        if args.dataset =="iemocap":
            utils.alignment_line_class_save_scatter(X, Y, args.save_analysis_path, 1)
        else:
            utils.alignment_line_class_meld_save_scatter(X, Y, args.save_analysis_path, 1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training_data")

    parser.add_argument(
        "--specific",
        type=str,
        default=True,
        choices=[True, False],
        help="whether to use a modality-specific model or not.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="meld",
        choices=["iemocap", "meld"],
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

    setting = "specific" if temp.specific else "agnostic"
    args = utils.get_config_args(parser, 'config/'+temp.dataset+'_'+setting+'.yaml', dataset=temp.dataset)

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
