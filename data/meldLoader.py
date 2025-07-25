from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class MELDDataset(Dataset):
    def __init__(self, path, indices=None):
        with open(path, "rb") as f:
            data = pickle.load(f)


        self.videoAudio = {k: data[4][k] for k in indices}  # videoText
        self.videoText = {k: data[3][k] for k in indices}  # videoAudio
        self.videoVisual = {k: data[5][k] for k in indices}  # videoVisual
        self.videoLabels = {k: data[2][k] for k in indices}  # videoLabels
        self.videoSpeakers = {k: data[1][k] for k in indices}  # videoSpeakers

        self.keys = list(self.videoLabels.keys())  # 정렬된 key 리스트
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(self.videoText[vid]),
            torch.FloatTensor(self.videoVisual[vid]),
            torch.FloatTensor(self.videoAudio[vid]),
            torch.FloatTensor(self.videoSpeakers[vid]),
            torch.FloatTensor([1] * len(self.videoLabels[vid])),  # Mask
            torch.LongTensor(self.videoLabels[vid]),
            vid,
        )

    def __len__(self):
        return self.len

def get_MELD_loaders(batch_size, data_path, num_workers=0, pin_memory=False):
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    all_keys = list(data[2].keys())  # videoLabels의 key 사용
    train_len, valid_len = 1008, 144

    # 인덱스 생성
    train_idx = all_keys[:train_len]
    valid_idx = all_keys[train_len:train_len + valid_len]
    test_idx = all_keys[train_len + valid_len:]
    all_idx = all_keys

    # 필요한 데이터만 포함한 새로운 MELDDataset 생성
    train_dataset = MELDDataset('/home/neuroai/users/dhkim/Multimodal-Graphormer/data/meld/MELD_features_raw1.pkl', train_idx)
    valid_dataset = MELDDataset('/home/neuroai/users/dhkim/Multimodal-Graphormer/data/meld/MELD_features_raw1.pkl', valid_idx)
    test_dataset = MELDDataset('/home/neuroai/users/dhkim/Multimodal-Graphormer/data/meld/MELD_features_raw1.pkl', test_idx)
    all_dataset = MELDDataset('/home/neuroai/users/dhkim/Multimodal-Graphormer/data/meld/MELD_features_raw1.pkl', all_idx)

    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn,
                             num_workers=num_workers, pin_memory=pin_memory)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, all_loader

def collate_fn(batch):
    """ 배치 데이터를 Tensor로 변환 """
    videoText, videoVisual, videoAudio, videoSpeakers, mask, videoLabels, vid = zip(*batch)

    return {
        "videoAudio": pad_sequence(videoAudio, batch_first=True),
        "videoText": pad_sequence(videoText, batch_first=True),
        "videoVisual": pad_sequence(videoVisual, batch_first=True),
        "videoLabels": pad_sequence(videoLabels, batch_first=True, padding_value=-1),  # 라벨은 -1로 패딩
        "videoSpeakers": pad_sequence(videoSpeakers, batch_first=True),
        "mask": pad_sequence(mask, batch_first=True),
        "vid": vid,
    }
