import argparse
import utils
import random
import numpy as np
import os
from data import *
# --- (get_labels, remove_indices 함수는 그대로 사용) ---

def get_labels(data):
    label = list()
    for d in data:
        label.append(d)
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
    final_sample_data = [dataset[i] for i in sample_indices if i in dataset.keys()]

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
    args.data = os.path.join(pretraining_data_dir,"backup", "data_" + args.dataset + ".pkl")

    total_dataset = utils.load_pkl(args.data)

    # --- 데이터 분할 ---
    num_to_sample_train = 1152
    num_to_sample_dev = 0
    num_to_sample_test = 280

    dataset = total_dataset[2]
    total_indicies = [i for i in range(len(dataset.keys()))]

    new_dataset = {}
    train_dialogue_idx = [227, 1234, 938, 340, 1037, 496, 546, 432, 3, 929, 1075, 450, 832, 833, 246, 882, 597, 997, 1274, 743, 1370, 804, 971, 174, 1161, 495, 374, 630, 1014, 987, 461, 277, 587, 127, 1100, 610, 58, 46, 1049, 177, 224, 454, 315, 186, 1065, 1181, 1393, 1201, 393, 1239, 1145, 460, 90, 1231, 119, 692, 1338, 475, 1318, 1128, 514, 1238, 679, 1273, 568, 1087, 138, 548, 2, 38, 988, 79, 98, 383, 196, 397, 851, 1344, 995, 1146, 595, 1053, 64, 1096, 30, 395, 336, 1108, 774, 131, 71, 70, 1335, 26, 1313, 1298, 606, 1175, 457, 1430, 840, 620, 1162, 571, 813, 59, 567, 54, 1199, 947, 1390, 1024, 1248, 1139, 1173, 888, 413, 1352, 211, 632, 427, 1127, 955, 545, 1257, 870, 298, 613, 417, 829, 631, 1125, 1310, 555, 1050, 989, 559, 920, 480, 1420, 81, 749, 1042, 1110, 655, 195, 1418, 695, 243, 519, 867, 621, 68, 894, 416, 680, 373, 1210, 544, 1398, 74, 1080, 876, 762, 981, 1033, 824, 65, 1172, 1068, 238, 721, 1427, 78, 504, 672, 233, 999, 1193, 217, 659, 1362, 285, 1268, 727, 242, 1254, 896, 934, 165, 1422, 673, 171, 1006, 1399, 1259, 1320, 16, 1023, 363, 852, 732, 100, 1153, 507, 1244, 167, 1221, 906, 358, 510, 515, 444, 112, 202, 970, 257, 105, 784, 566, 1281, 201, 1064, 474, 860, 893, 19, 953, 94, 230, 267, 1230, 532, 372, 37, 39, 1192, 643, 1264, 1134, 158, 1018, 986, 998, 584, 772, 685, 1347, 99, 691, 472, 602, 1402, 754, 302, 283, 1360, 436, 1079, 452, 134, 197, 927, 570, 977, 1417, 423, 405, 470, 715, 1028, 623, 1160, 618, 200, 209, 1109, 933, 1384, 1025, 1349, 702, 6, 864, 535, 572, 1198, 289, 1017, 687, 116, 740, 398, 983, 509, 1155, 485, 828, 121, 1062, 1246, 1182, 1021, 220, 1211, 608, 354, 471, 27, 1311, 199, 1119, 348, 160, 415, 22, 1357, 346, 561, 1138, 694, 1288, 619, 35, 437, 1036, 252, 1382, 1040, 1176, 826, 465, 915, 1007, 1364, 814, 1136, 846, 476, 1339, 533, 251, 181, 446, 908, 1381, 531, 720, 1377, 614, 1178, 414, 960, 69, 539, 819, 836, 585, 31, 258, 1209, 710, 447, 793, 198, 1276, 1048, 776, 294, 612, 1373, 1269, 656, 148, 665, 830, 773, 396, 984, 1194, 701, 1252, 487, 816, 848, 1391, 850, 948, 856, 152, 1423, 319, 222, 768, 1293, 494, 863, 1061, 456, 518, 1137, 1163, 187, 103, 1002, 1300, 32, 590, 838, 1144, 780, 1312, 1001, 600, 163, 1010, 1212, 782, 807, 262, 376, 654, 189, 1258, 605, 534, 276, 1106, 92, 943, 1243, 502, 166, 1282, 809, 645, 603, 329, 1358, 683, 239, 232, 93, 210, 345, 902, 763, 1271, 228, 689, 936, 125, 964, 85, 50, 808, 1410, 1191, 500, 958, 60, 932, 422, 1073, 563, 565, 269, 1317, 1235, 693, 892, 949, 900, 5, 895, 513, 420, 443, 1015, 1117, 799, 24, 255, 520, 1222, 1152, 1315, 1256, 88, 1206, 13, 355, 648, 1380, 706, 1323, 1005, 1085, 11, 1121, 841, 299, 1101, 1237, 540, 161, 1331, 76, 965, 626, 551, 765, 499, 1, 1328, 527, 526, 126, 91, 1112, 842, 878, 1041, 1421, 926, 923, 524, 43, 1056, 124, 1190, 72, 157, 1174, 483, 10, 1093, 593, 1224, 722, 916, 1202, 62, 935, 352, 1295, 493, 205, 419, 769, 1270, 1404, 657, 1047, 1032, 473, 28, 1355, 300, 1027, 305, 1063, 954, 1346, 1342, 1147, 364, 1376, 1388, 558, 9, 974, 1251, 1187, 241, 601, 738, 314, 284, 468, 1129, 25, 250, 135, 1038, 549, 411, 1180, 1169, 827, 359, 950, 638, 316, 287, 890, 429, 1394, 1405, 1428, 1203, 589, 104, 696, 883, 1022, 102, 719, 322, 1054, 698, 1219, 663, 1060, 1290, 918, 325, 512, 491, 77, 1401, 581, 709, 723, 1031, 1359, 1326, 1386, 810, 952, 1261, 798, 914, 834, 662, 394, 975, 1289, 96, 651, 154, 168, 550, 525, 424, 778, 635, 190, 1107, 622, 1070, 1365, 1120, 609, 748, 556, 453, 1124, 928, 80, 1217, 390, 1090, 117, 360, 1275, 139, 713, 641, 1133, 378, 1115, 133, 905, 1208, 1240, 1184, 1327, 910, 898, 1324, 271, 871, 978, 463, 913, 1387, 162, 707, 708, 388, 939, 825, 61, 688, 20, 240, 577, 669, 728, 1195, 919, 1067, 1213, 1167, 387, 861, 789, 430, 1026, 1084, 887, 245, 541, 14, 1154, 1353, 333, 903, 993, 270, 760, 244, 885, 445, 1429, 795, 615, 15, 144, 796, 538, 229, 560, 268, 1043, 292, 594, 761, 1332, 1008, 1051, 477, 547, 467, 86, 1130, 1348, 869, 922, 1204, 957, 1241, 488, 862, 1356, 879, 946, 1314, 764, 1044, 173, 791, 273, 193, 330, 84, 1413, 968, 123, 812, 110, 1227, 418, 1233, 1046, 604, 1166, 1012, 66, 1411, 1236, 788, 1366, 874, 1156, 1183, 145, 1188, 362, 291, 75, 184, 951, 328, 335, 213, 661, 516, 150, 392, 1378, 399, 849, 1286, 671, 802, 331, 847, 886, 1325, 164, 17, 18, 1406, 717, 466, 408, 991, 309, 592, 739, 742, 1292, 1069, 1039, 323, 755, 235, 1135, 675, 136, 182, 391, 290, 1308, 234, 384, 1148, 403, 478, 87, 1089, 369, 1220, 129, 442, 664, 1103, 172, 433, 557, 368, 223, 744, 844, 1123, 56, 497, 966, 498, 381, 746, 1097, 343, 1140, 627, 188, 1350, 668, 350, 501, 1013, 1322, 684, 1077, 543, 1207, 386, 389, 23, 490, 370, 872, 254, 481, 1345, 122, 523, 1228, 877, 1415, 297, 1150, 44, 800, 1059, 307, 462, 1131, 1319, 357, 1086, 831, 146, 537, 956, 1099, 141, 1020, 253, 658, 1170, 1197, 1092, 969, 434, 1403, 216, 553, 0, 332, 155, 412, 1272, 575, 1232, 1030, 1280, 1159, 280, 1055, 783, 917, 730, 564, 281, 169, 256, 170, 1242, 118, 449, 288, 279, 1082, 308, 881, 705, 1316, 1126, 741, 855, 214, 1171, 1255, 676, 380, 4, 1307, 718, 296, 409, 899, 1400, 726, 455, 574, 839, 781, 576, 428, 875, 1113, 554, 1076, 837, 880, 1034, 1285, 1262, 248, 382, 529, 737, 1189, 479, 633, 275, 47, 1375, 451, 647, 57, 1279, 652, 1385, 356, 854, 1215, 644, 1186, 1351, 930, 1369, 517, 206, 327, 179, 790, 120, 901, 36, 115, 1296, 1009, 1151, 759, 12, 221, 1407, 912, 911, 431, 313, 435, 402, 599, 1165, 261, 1094, 1029, 1072, 1141, 530, 931, 218, 438, 1066, 750, 272, 617, 1277, 660, 940, 274, 1414, 1278, 1431, 1250, 1179, 379, 801, 704, 55, 1302, 439, 1265, 1245, 203, 486, 503, 637, 941, 640, 159, 1361, 1408, 318, 337, 140, 226, 578, 979, 703, 1095, 407, 48, 249, 113, 236, 747, 1098, 904, 303, 406, 1287, 1426, 1078, 865, 1321, 143, 1379, 591, 1205, 994, 247, 1374, 805, 1105, 745, 803, 1333, 598, 33, 338, 1168, 51, 569, 1354, 29, 697, 1083, 1294, 1058, 634, 980, 82, 1372, 317, 859, 1247, 787, 1383, 1334, 858, 650, 753, 306, 1102, 766, 326, 992, 818, 339, 1004, 962, 1074, 777, 973, 175, 751, 596, 670, 891, 624, 815, 301, 1329, 1416, 1000, 729]
    dev_dialogue_idx = []
    test_dialogue_idx = [41, 1424, 779, 976, 183, 579, 484, 440, 959, 53, 817, 375, 794, 1412, 1297, 1340, 1363, 1309, 937, 811, 690, 1284, 1111, 506, 1132, 823, 736, 1052, 361, 101, 21, 522, 752, 712, 1223, 114, 265, 1118, 1104, 1419, 1088, 797, 1225, 616, 63, 324, 1214, 45, 1142, 653, 1057, 353, 583, 528, 342, 153, 1371, 771, 401, 425, 1149, 185, 156, 1216, 295, 151, 1397, 176, 700, 219, 459, 725, 1341, 52, 1122, 505, 845, 1368, 349, 646, 562, 942, 1336, 204, 921, 404, 286, 1035, 677, 982, 792, 907, 1291, 212, 108, 735, 843, 266, 492, 1395, 990, 7, 649, 237, 192, 191, 142, 785, 1114, 1196, 925, 1091, 909, 111, 304, 366, 628, 334, 536, 1306, 147, 674, 489, 278, 310, 666, 1011, 1071, 682, 1003, 311, 83, 699, 611, 34, 178, 873, 1249, 377, 521, 1330, 734, 1389, 1116, 588, 282, 639, 1157, 1396, 853, 884, 985, 580, 426, 347, 667, 1081, 1305, 1019, 607, 42, 1367, 1267, 1143, 821, 1337, 128, 95, 194, 678, 231, 972, 344, 1343, 73, 944, 1301, 482, 681, 967, 508, 40, 716, 264, 552, 215, 1016, 924, 866, 542, 714, 733, 400, 1226, 367, 945, 259, 1425, 757, 260, 820, 586, 320, 410, 786, 582, 1218, 1045, 385, 441, 822, 1283, 293, 756, 1177, 1229, 806, 1299, 371, 97, 625, 573, 636, 67, 207, 341, 464, 1303, 686, 8, 458, 312, 629, 1200, 137, 1266, 1260, 149, 897, 642, 857, 511, 1253, 1158, 731, 132, 889, 1263, 1409, 106, 448, 421, 351, 365, 963, 868, 711, 1392, 130, 1185, 767, 469, 996, 961, 1304, 180, 758, 109, 89, 263, 208, 321, 107, 775, 1164, 225, 49, 724, 835, 770]
    new_dataset = total_dataset
    new_dataset[7] = train_dialogue_idx
    new_dataset[8] = dev_dialogue_idx
    new_dataset[9] = test_dialogue_idx

    new_data = os.path.join(pretraining_data_dir, "data_" + args.dataset + ".pkl")

    utils.save_pkl(new_dataset, new_data)
    print(f"total number of dialogues: {len(train_dialogue_idx) + len(dev_dialogue_idx) + len(test_dialogue_idx)}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="pretraining_data")

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

    settings = "specific" if temp.specific else "agnostic"
    args = utils.get_config_args(parser, 'config/'+temp.dataset+'_'+settings+'.yaml', dataset=temp.dataset)

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

