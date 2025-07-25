import os
import sys
import random
import logging
import yaml
import torch
import pickle

from datetime import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logging.basicConfig(force=True, level=logging.INFO)

def set_seed(seed):
    """Sets random seed everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # use determinisitic algorithm
    print("Seed set", seed)




def plot_and_save_confusion_matrix(golds, preds, class_labels:dict, save_path="confusion_matrix.png",
                                   fontsize=12):
    """
    Normalized Confusion Matrix를 생성하고 타이트한 형태로 저장하는 함수 (폰트 조정 가능)

    Args:
        golds (list or np.array): 실제 레이블 (정답)
        preds (list or np.array): 예측 레이블
        class_labels (list): 클래스 이름 리스트
        save_path (str): 저장할 이미지 파일 경로
        fontsize (int): 폰트 크기
    """
    # ✅ Confusion Matrix 계산 (normalize='true'를 사용하여 행 단위 정규화)
    cm = confusion_matrix(golds, preds, normalize='true')

    # ✅ Seaborn 스타일 설정
    sns.set(font_scale=1.2)  # 전체 폰트 크기 스케일링

    # ✅ 시각화 (Seaborn을 이용한 Confusion Matrix Plot)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels.keys(), yticklabels=class_labels.keys(),
                     annot_kws={"size": fontsize})  # ✅ 셀 내부 숫자 폰트 크기 조정

    # ✅ 축 라벨 및 제목 폰트 크기 설정
    plt.xlabel("Predicted Labels", fontsize=fontsize)
    plt.ylabel("True Labels", fontsize=fontsize)
    plt.title("Normalized Confusion Matrix", fontsize=fontsize)

    # ✅ X/Y 눈금 폰트 크기 조정
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    # ✅ Confusion Matrix 타이트하게 저장 (bbox_inches="tight" 추가)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()  # 화면에 출력
    plt.close()  # 메모리 절약을 위해 close()
    print(f"✅ Confusion Matrix saved to {save_path} (tight box format)")



def get_logger(filepath: str, level=logging.INFO):
    logger = logging.getLogger(__name__)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    fileHandler = logging.FileHandler(filepath)
    streamHandler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        fmt='[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s'
    )
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger


def get_config_args(parser, yaml_file_path, dataset):
    # Load YAML file
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)[dataset]

    # Add arguments with defaults from YAML
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                parser.add_argument(f'--{key}_{sub_key}', type=type(sub_value), default=sub_value)
        elif isinstance(value, list):
            parser.add_argument(f'--{key}', type=str, nargs='+', default=value)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)

    # Parse arguments
    args = parser.parse_args()

    return args


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def make_route(dir_path, file_name=None):
    # Full path for the directory
    absolute_path = os.path.join(os.getcwd(), dir_path)

    # Check if the directory exists, create it if it doesn't
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    # only for making directory
    if file_name is None:
        return

    # Full path for the file inside the directory
    file_path = os.path.join(absolute_path, file_name)

    # Check if the file already exists
    if os.path.exists(file_path):
        # Get the current date and time
        current_datetime = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
        # Define the new filename with the current date and time
        title, extension = os.path.splitext(file_name)
        new_file_name = f'{title}-backup-{current_datetime}-{extension}'
        # Rename the existing file
        new_file_path = os.path.join(absolute_path, new_file_name)
        os.rename(file_path, new_file_path)

    # Create a new file (or open the file if it somehow already exists) and write something to it
    f = open(file_path, 'w')
    f.close()

    return


def plot_and_save_loss(train_losses, val_losses, test_losses, filename):
    """
    Generates a line graph comparing training and validation losses over epochs and saves the figure to a file.

    Parameters:
    - train_losses (list of float): The training losses for each epoch.
    - val_losses (list of float): The validation losses for each epoch.
    - filename (str): The name of the file to save the plot. Defaults to 'loss_comparison.png'.

    Returns:
    - None
    """
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.plot(epochs, test_losses, label = 'Test Loss')

    plt.title('Training vs Validation vs Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    plt.show()

    print(f"Plot saved as {filename}")