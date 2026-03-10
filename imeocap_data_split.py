import argparse
import utils
import random
import numpy as np
import os

# --- (get_labels, remove_indices 함수는 그대로 사용) ---

def get_labels(data):
    label = list()
    for d in data:
        label.append(d['labels'])
    return label


# --- get_samples 함수 수정 ---
# 단일 결과를 반환하도록 수정하고, num_classes를 인자로 받습니다.
def get_samples(dataset, num_classes):
    # population_indices = list(range(len(dataset)))
    #
    # # 인덱스를 랜덤하게 샘플링
    # sample_indices = random.sample(population_indices, num_sample)
    #
    # 샘플링된 인덱스로 실제 데이터 가져오기

    # 라벨 분포 및 표준편차 계산
    sample_labels = get_labels(dataset)
    sample_flat = [label for sublist in sample_labels for label in sublist]
    sample_counts = np.bincount(np.array(sample_flat), minlength=num_classes)
    std = np.std(sample_counts)

    # 샘플링된 인덱스, 라벨 분포, 표준편차를 반환
    return None, sample_counts, std


# --- main 함수 수정 ---
def main(args):
    utils.set_seed(args.seed)

    # --- 데이터 로드 (기존 코드와 동일) ---
    print("Load pretraining dataset... Name: " + args.dataset)

    pretraining_data_dir = os.path.join(os.getcwd(), args.data_dir_path, args.dataset)
    args.data = os.path.join(pretraining_data_dir, "data_" + args.dataset + ".pkl")

    dataset = utils.load_pkl(args.data)

    total_dataset = dataset['train'] + dataset['dev'] + dataset['test']

    new_dataset = {}
    train_dialogue_idx = [82, 14, 35, 1, 69, 13, 6, 16, 105, 45, 0, 83, 102, 67, 85, 90, 51, 104, 112, 47, 24, 42, 81, 100, 89, 122, 41, 123, 93, 128, 33, 63, 113, 34, 39, 101, 110, 31, 18, 17, 136, 142, 80, 15, 73, 68, 19, 148, 27, 86, 56, 141, 146, 87, 62, 2, 98, 28, 59, 133, 129, 50, 97, 135, 143, 96, 21, 137, 140, 132, 10, 126, 70, 7, 55, 79, 116, 130, 94, 92, 75, 49, 8, 40, 149, 134, 32, 84, 12, 60, 11, 139, 118, 77, 131, 109, 29, 72, 52, 30, 145, 91, 3, 106, 5, 120, 37, 107]

    train_dialogue = []
    for idx in train_dialogue_idx:
        train_dialogue.append(total_dataset[idx])

    dev_dialogue_idx = [42, 51, 56, 100, 60, 93, 37, 80, 50, 29, 84, 47]

    dev_dialogue = []
    for idx in dev_dialogue_idx:
        dev_dialogue.append(total_dataset[idx])

    test_dialogue_idx = [144, 78, 150, 53, 22, 108, 124, 74, 4, 95, 138, 20, 115, 38, 46, 57, 25, 111, 58, 23, 44, 117, 64, 147, 88, 26, 119, 121, 125, 9, 43]

    test_dialogue = []
    for idx in test_dialogue_idx:
        test_dialogue.append(total_dataset[idx])

    new_dataset['train'] = train_dialogue
    new_dataset['dev'] = dev_dialogue
    new_dataset['test'] = test_dialogue

    utils.save_pkl(new_dataset, args.data)
    print(f"total number of dialogues: {len(train_dialogue_idx) + len(dev_dialogue_idx) + len(test_dialogue_idx)}")


    _, train_sample_count, train_std = get_samples(train_dialogue, num_classes=args.num_classes)
    _, dev_sample_count, dev_std = get_samples(dev_dialogue, num_classes=args.num_classes)
    _, test_sample_count, test_std = get_samples(test_dialogue, num_classes=args.num_classes)

    print(f"Train sample count: {train_sample_count} ({train_std})")
    print(f"Dev sample count: {dev_sample_count} ({dev_std})")
    print(f"Test sample count: {test_sample_count} ({test_std})")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="iemocap",
        choices=["iemocap", "iemocap_4", "mosei", "meld"],
        help="Dataset name."
    )
    # parser.add_argument(
    #     "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
    # )

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in UnimodalEncoder.py"""
    # parser.add_argument(
    #     "--modalities",
    #     type=str,
    #     default="atv",
    #     # required=True,
    #     choices=["a", "t", "v", "at", "tv", "av", "atv"],
    #     help="Modalities",
    # )

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
