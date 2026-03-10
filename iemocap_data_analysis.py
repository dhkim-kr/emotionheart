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


def remove_indices(source_list, indices_to_remove):
    indices_to_remove_set = set(indices_to_remove)
    result_list = [
        value for i, value in enumerate(source_list)
        if i not in indices_to_remove_set
    ]
    return result_list


# --- get_samples 함수 수정 ---
# 단일 결과를 반환하도록 수정하고, num_classes를 인자로 받습니다.
def get_samples(dataset, population_indices, num_sample, num_classes):

    # 인덱스를 랜덤하게 샘플링
    sample_indices = random.sample(population_indices, num_sample)

    # 샘플링된 인덱스로 실제 데이터 가져오기
    final_sample_data = [dataset[i] for i in sample_indices]

    # 라벨 분포 및 표준편차 계산
    sample_labels = get_labels(final_sample_data)

    sample_flat = [label for sublist in sample_labels for label in sublist]
    sample_counts = np.bincount(np.array(sample_flat), minlength=num_classes)
    std = np.std(sample_counts)

    # 샘플링된 인덱스, 라벨 분포, 표준편차를 반환
    return sample_indices, sample_counts, std


# --- main 함수 수정 ---
def main(args):
    utils.set_seed(args.seed)

    # --- 데이터 로드 (기존 코드와 동일) ---
    print("Load pretraining dataset... Name: " + args.dataset)

    pretraining_data_dir = os.path.join(os.getcwd(), args.data_dir_path, args.dataset)
    args.data = os.path.join(pretraining_data_dir, "data_" + args.dataset + ".pkl")

    dataset = utils.load_pkl(args.data)

    # 전체 데이터셋을 미리 합쳐놓음
    total_dataset = dataset['train'] + dataset['dev'] + dataset['test']

    # --- 데이터 분할 ---
    num_to_sample_train = 108
    num_to_sample_dev = 12
    num_to_sample_test = 31

    total_indicies = [i for i in range(len(total_dataset))]


    # 최대한 균형적인 데이터 스플릿 구성 찾기
    print("\n Newly Sample pretrain, finetune, dev, test datasets")

    # 1. 최적의 결과를 저장할 변수 초기화
    best_config = None
    min_total_std = float('inf')  # 표준편차 합계의 최소값을 저장 (초기값은 무한대)


    # 백만 번 반복
    for i in range(1000000):
        #1. Test 셋 샘플링
        test_indices, test_counts, test_std = get_samples(total_dataset, total_indicies, num_to_sample_test,
                                                                   args.num_classes)

        # 2. Test 셋을 제외한 나머지 데이터 생성
        remaining_after_test= remove_indices(total_indicies, test_indices)

        # 3. 남은 데이터에서 Dev 셋 샘플링
        dev_indices, dev_counts, dev_std = get_samples(total_dataset, remaining_after_test, num_to_sample_dev, args.num_classes)

        # 4. Dev 셋을 제외한 나머지 데이터 생성 (이것이 Pretrain 셋이 됨)
        remaining_after_dev = remove_indices(remaining_after_test, dev_indices)

        # 5. Train 셋 정보 계산
        train_indices, train_counts, train_std = get_samples(total_dataset, remaining_after_dev, num_to_sample_train, args.num_classes)

        # --- 최적의 구성인지 판단 ---
        # 6. 이번 반복의 총 표준편차 계산 (train, dev, test의 std 합)
        # current_total_std = dev_std + test_std

        # # 1. Train 셋 샘플링
        # train_indices, train_counts, train_std = get_samples(total_dataset, num_to_sample_train, args.num_classes)
        #
        # # 2. Train 셋을 제외한 나머지 데이터 생성
        # remaining_after_train = remove_indices(total_dataset, train_indices)
        #
        # # 3. 남은 데이터에서 Test 셋 샘플링
        # test_indices, test_counts, test_std = get_samples(remaining_after_train, num_to_sample_test,
        #                                                            args.num_classes)
        #
        # # 4. Test 셋을 제외한 나머지 데이터 생성 (이것이 Dev 셋이 됨)
        # remaining_after_test = remove_indices(remaining_after_train, test_indices)
        #
        # # 5. Dev 셋 정보 계산
        # # dev_indices는 전체 데이터셋 기준이 아닌, 나머지 데이터 기준이므로 직접 계산
        # dev_indices, dev_counts, dev_std = get_samples(remaining_after_test, num_to_sample_dev, args.num_classes)
        #
        # # --- 최적의 구성인지 판단 ---
        # # 2. 이번 반복의 총 표준편차 계산 (train, dev, test의 std 합)
        # current_total_std = train_std + dev_std + test_std

        current_total_std = train_std #+test_std
        best_idx = [1,3,4]
        worst_idx = [2,5]
        best_counts = test_counts[best_idx]
        best = np.array(best_counts).sum()
        # worst = np.array(test_counts[worst_idx]).sum()

        worst1 = np.array(test_counts[2]).sum()
        worst2 = np.array(test_counts[5]).sum()

        # standard = np.sum(train_counts)/np.sum(test_counts)
        # ratio_std = np.std(train_counts / test_counts)
        # print(f"train counts: {train_counts}, test counts: {test_counts}, standard: {standard:.4f}")
        # 7. 현재까지의 최소 표준편차보다 작으면, 이번 구성을 '최고의 구성'으로 저장
        if current_total_std < min_total_std and worst1<300 and worst2 <300 and best_counts[0]>250 and best_counts[1]>250 and best_counts[2]>250: #and ratio_std < standard:
            min_total_std = current_total_std
            best_config = {
                "iteration": i,
                "train_indices": train_indices,
                "dev_indices": dev_indices,
                "test_indices": test_indices,
                "train_counts": train_counts,
                "dev_counts": dev_counts,
                "test_counts": test_counts,
                "train_std": train_std,
                "dev_std": dev_std,
                "test_std": test_std,
                "total_std": min_total_std
                # 필요하다면 인덱스도 저장: "train_indices": train_indices
            }
            # 새로운 최고 기록이 나올 때마다 출력
            print(f"New Best Found at iteration {i}! Total STD: {min_total_std:.4f}")#, ratio STD: {ratio_std:.4f}")
            # 8. 백만 번 반복이 끝난 후, 최종적으로 찾은 최고의 구성 출력
            print("\n--- Final Best Configuration ---")
            if best_config:
                print(f"Best iteration: {best_config['iteration']}")
                print(f"Minimum Total STD: {best_config['total_std']:.4f}")
                print(f"Train indices: {best_config['train_indices']}")
                print(f"Dev indices: {best_config['dev_indices']}")
                print(f"Test indices: {best_config['test_indices']}")
                print(
                    f"Train Counts: {best_config['train_counts']} (Sum: {np.sum(best_config['train_counts'])}), (STD: {best_config['train_std']:.4f})")
                print(
                    f"Dev Counts: {best_config['dev_counts']} (Sum: {np.sum(best_config['dev_counts'])}), (STD: {best_config['dev_std']:.4f})")
                print(
                    f"Test Counts: {best_config['test_counts']} (Sum: {np.sum(best_config['test_counts'])}), (STD: {best_config['test_std']:.4f})")
            else:
                print("No valid configuration was found.")

    # 8. 백만 번 반복이 끝난 후, 최종적으로 찾은 최고의 구성 출력
    print("\n--- Final Best Configuration ---")
    if best_config:
        print(f"Best iteration: {best_config['iteration']}")
        print(f"Minimum Total STD: {best_config['total_std']:.4f}")
        print(f"Train indices: {best_config['train_indices']}")
        print(f"Dev indices: {best_config['dev_indices']}")
        print(f"Test indices: {best_config['test_indices']}")
        print(f"Train Counts: {best_config['train_counts']} (Sum: {np.sum(best_config['train_counts'])}), (STD: {best_config['train_std']:.4f})")
        print(f"Dev Counts: {best_config['dev_counts']} (Sum: {np.sum(best_config['dev_counts'])}), (STD: {best_config['dev_std']:.4f})")
        print(f"Test Counts: {best_config['test_counts']} (Sum: {np.sum(best_config['test_counts'])}), (STD: {best_config['test_std']:.4f})")
    else:
        print("No valid configuration was found.")

    # 9. 최적의 결과를 저장할 변수 초기화
    best_config = None
    min_finetune_train_std = float('inf')  # 표준편차 합계의 최소값을 저장 (초기값은 무한대)
    #
    # for i in range(1000000):
    #     finetune_train_indices, finetune_train_counts, finetune_train_std = get_samples(remaining_after_dev, num_to_sample_train_fintune, args.num_classes)
    #
    #     # 10. 현재까지의 최소 표준편차보다 작으면, 이번 구성을 '최고의 구성'으로 저장
    #     if finetune_train_std < min_finetune_train_std:
    #         min_finetune_train_std = finetune_train_std
    #         best_config = {
    #             "iteration": i,
    #             "finetune_train_indices": finetune_train_indices,
    #             "finetune_train_counts": finetune_train_counts,
    #             "min_finetune_train_std": min_finetune_train_std
    #         }
    # if best_config:
    #     print(f"Best iteration: {best_config['iteration']}")
    #     print(f"Minimum Total STD: {best_config['min_finetune_train_std']:.4f}")
    #     print(f"Finetune Train indices: {best_config['finetune_train_indices']}")
    #     print(f"Finetune Counts: {best_config['finetune_train_counts']} ({best_config['min_finetune_train_std']:.4f})")
    # else:
    #     print("No valid configuration was found.")

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
#
#
# def main(args):
#     utils.set_seed(args.seed)
#
#     # load data
#     log.info("Load pretraining dataset... Name: " + args.dataset)
#     pretraining_data_dir = os.path.join(os.getcwd(), args.data_dir_path, args.dataset)
#     args.data = os.path.join(pretraining_data_dir, "data_" + args.dataset + ".pkl")
#
#     dataset = utils.load_pkl(args.data)
#
#     total_dataset = []
#     temp = None
#
#     for i in range(1000000):
#         # 뽑고 싶은 개수
#         num_to_sample_train = 108
#         num_to_sample_dev = 12
#         num_to_sample_test = 31
#
#         total_dataset.extend(dataset['train'])
#         total_dataset.extend(dataset['dev'])
#         total_dataset.extend(dataset['test'])
#
#         # 151개 리스트에서 랜덤하게 108개를 샘플링
#         train_random_sample_idx, train_sample_counts, train_std = get_samples(total_dataset, num_to_sample_train)
#
#         temp = remove_indices(total_dataset, train_random_sample_idx)
#         # 나머지 43개 중 31 개 샘플링
#         test_random_sample_idx, test_sample_counts, test_std = get_samples(temp, num_to_sample_test)
#
#         temp = remove_indices(temp, num_to_sample_dev)
#         # 나머지 12개 중 12 개 샘플링
#         dev_random_sample_idx, dev_sample_counts, dev_std = get_samples(temp, num_to_sample_dev)
#
#         print(f"{i}-th iteration\n *Train -- random samples: {train_random_sample_idx} \n Sample Label Counts: {train_sample_counts}- STD: {train_std}\n *Test -- random samples: {test_random_sample_idx} \n Sample Label Counts: {test_sample_counts}- STD: {test_std}\n *Dev -- random samples: {dev_random_sample_idx} \n Sample Label Counts: {dev_sample_counts}- STD: {dev_std}\n")
#
#     min_std, argmin_std = np.min(stds), np.argmin(stds)
#     print(f"Best Results: {argmin_std}-th iteration -- random samples: {sample_list[argmin_std]} \n Sample Label Counts: {sample_count_list[argmin_std]}- STD: {min_std}\n\n")
#
# def get_labels(data):
#     label = list()
#     for d in data:
#         label.append(d['labels'])
#     return label
#
# def get_samples(dataset, num_sample):
#     indices = list(range(len(dataset)))
#     stds = []
#     sample_count_list = []
#     sample_list = []
#
#     random_sample = random.sample(indices, num_sample)
#     final_sample = [dataset[i] for i in random_sample]
#
#     sample_label = get_labels(final_sample)
#     sample_flat = [label for sublist in sample_label for label in sublist]
#     sample_counts = np.bincount(np.array(sample_flat), minlength=args.num_classes)
#
#     std = np.std(sample_counts)
#
#     sample_list.append(random_sample)
#     sample_count_list.append(sample_counts)
#     stds.append(std)
#
#     return sample_list, sample_count_list, stds
#
# def remove_indices(source_list, indices_to_remove):
#     # 1. 제거할 인덱스들을 set으로 변환 (검색 속도 향상)
#     indices_to_remove_set = set(indices_to_remove)
#
#     # 2. 리스트 내포를 사용해 새로운 리스트 생성
#     # enumerate로 각 아이템의 인덱스(i)를 가져와서,
#     # 그 인덱스가 제거할 set에 없는 경우에만 최종 리스트에 포함
#     result_list = [
#         value for i, value in enumerate(source_list)
#         if i not in indices_to_remove_set
#     ]
#     return result_list
#     # train_label = get_labels(dataset['train'])
#     # dev_label = get_labels(dataset['dev'])
#     # test_label = get_labels(dataset['test'])
#     #
#     # # --- 각 데이터셋별로 개수 계산 ---
#     #
#     # # 1. Train set
#     # train_flat = [label for sublist in train_label for label in sublist]
#     # train_counts = np.bincount(np.array(train_flat), minlength=args.num_classes)
#     #
#     # # 2. Dev set
#     # dev_flat = [label for sublist in dev_label for label in sublist]
#     # dev_counts = np.bincount(np.array(dev_flat), minlength=args.num_classes)
#     #
#     # # 3. Test set
#     # test_flat = [label for sublist in test_label for label in sublist]
#     # test_counts = np.bincount(np.array(test_flat), minlength=args.num_classes)
#     #
#     # # --- 결과 출력 ---
#     # print(f"Train Label Counts: {train_counts}")
#     # print(f"Dev Label Counts:   {dev_counts}")
#     # print(f"Test Label Counts:  {test_counts}")
#
#
#
#
#
#
#
#     # if args.dataset == "iemocap" or args.dataset == "iemocap_4":
#     #     dataset = utils.load_pkl(args.data)
#     #     graph_trainset_file = os.path.join(pretraining_data_dir, "graph_trainset.pkl")
#     #
#     #     if not os.path.exists(graph_trainset_file):
#     #         pretrainset = gdt.iemocap_4_graphDataset(dataset["train"], 'train', args)
#     #         utils.save_pkl(pretrainset, graph_trainset_file)
#     #     pretrainset = utils.load_pkl(graph_trainset_file)
#     #
#     #     args.n_max_utterances = pretrainset.n_max_utterances
#     #     args.n_max_speakers = pretrainset.n_max_speakers
#     #
#     #
#     # if args2.dataset == "iemocap_4" or args2.dataset == "iemocap":
#     #     finetuning_data = utils.load_pkl(args2.data)
#     #     graph_trainset_file = os.path.join(finetuning_data_dir, "graph_trainset.pkl")
#     #     graph_devset_file = os.path.join(finetuning_data_dir, "graph_devset.pkl")
#     #     graph_testset_file = os.path.join(finetuning_data_dir, "graph_testset.pkl")
#     #
#     #     if not os.path.exists(graph_trainset_file):
#     #         trainset = gdt.iemocap_4_graphDataset(finetuning_data["train"], 'train', args2)
#     #         utils.save_pkl(trainset, graph_trainset_file)
#     #     trainset = utils.load_pkl(graph_trainset_file)
#     #
#     #     # args.num_nodes = trainset.n_max_utterances * trainset.n_modalities  # with virtual node (graph token, e.g., CLS token in BERT)
#     #     args2.n_max_utterances = trainset.n_max_utterances
#     #     args2.n_max_speakers = trainset.n_max_speakers
#     #
#     #     if not os.path.exists(graph_devset_file):
#     #         devset = gdt.iemocap_4_graphDataset(finetuning_data["dev"], 'dev', args2)
#     #         utils.save_pkl(devset, graph_devset_file)
#     #     devset = utils.load_pkl(graph_devset_file)
#     #
#     #     if not os.path.exists(graph_testset_file):
#     #         testset = gdt.iemocap_4_graphDataset(finetuning_data["test"], 'test', args2)
#     #         utils.save_pkl(testset, graph_testset_file)
#     #     testset = utils.load_pkl(graph_testset_file)
#
#
#
# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(description="pretraining_data")
#
#     parser.add_argument(
#         "--dataset",
#         type=str,
#         # required=True,
#         default="iemocap",
#         choices=["iemocap", "iemocap_4", "mosei", "meld"],
#         help="Dataset name."
#     )
#     # parser.add_argument(
#     #     "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
#     # )
#
#     # Modalities
#     """ Modalities effects:
#         -> dimentions of input vectors in dataset.py
#         -> number of heads in transformer_conv in UnimodalEncoder.py"""
#     # parser.add_argument(
#     #     "--modalities",
#     #     type=str,
#     #     default="atv",
#     #     # required=True,
#     #     choices=["a", "t", "v", "at", "tv", "av", "atv"],
#     #     help="Modalities",
#     # )
#
#     parser.add_argument(
#         "--relation_type",
#         default="eam",
#         choices=["e", "ea", "eam"],
#         help="Choose relation contruct type. e: interlocuter, a: intralocuter, m: intermodality",
#     )
#
#     parser.add_argument(
#         "--optimizer",
#         default="adamw",
#         choices=["sgd","rmsprop","adam","adamw"],
#         help="Choose optimizer",
#     )
#
#     temp = parser.parse_args()
#
#     args = utils.get_config_args(parser, temp.dataset+'_pretrain.yaml', dataset=temp.dataset)
#
#     args.num_edges = len(args.relation_type)
#     args.num_degree = len(args.relation_type)
#     if args.relation_type == "eam" and args.modalities == "atv":
#         args.num_degree += 1
#
#     args.ffn_embed_dim = args.encoder_embed_dim * args.ffn_embed_scaler
#
#     args.dataset_embedding_dims = {
#         "iemocap": {
#             "a": 100,
#             "t": 768,
#             "v": 512,
#         },
#         "iemocap_4": {
#             "a": 100,
#             "t": 768,
#             "v": 512,
#         },
#         "mosei": {
#             "a": 80,
#             "t": 768,
#             "v": 35,
#         },
#         "meld": {
#             "a":300,
#             "t":600,
#             "v":342
#         }
#     }
#
#
#     log.debug(args)
#
#     main(args)

