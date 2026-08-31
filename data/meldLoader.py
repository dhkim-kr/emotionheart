import os
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

class MELDDataset(Dataset):
    def __init__(self, path, indices=None):
        with open(path, "rb") as f:
            data = pickle.load(f)


        self.videoAudio = {k: data[4][k] for k in indices if k in data[4]}  # videoText
        self.videoText = {k: data[3][k] for k in indices if k in data[3]}  # videoAudio
        self.videoVisual = {k: data[5][k] for k in indices if k in data[5]}  # videoVisual
        self.videoLabels = {k: data[2][k] for k in indices if k in data[2]}  # videoLabels
        self.videoSpeakers = {k: data[1][k] for k in indices if k in data[1]}  # videoSpeakers

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

    # 인덱스 생성
    train_idx = data[7]
    valid_idx = data[8]
    test_idx = data[9]
    all_idx = all_keys

    # data_path (data_meld.pkl) provides the split indices; the raw multimodal
    # features live in MELD_features_raw1.pkl next to it.
    features_path = os.path.join(os.path.dirname(data_path), 'MELD_features_raw1.pkl')

    train_dataset = MELDDataset(features_path, train_idx)
    if len(valid_idx) > 0:
        valid_dataset = MELDDataset(features_path, valid_idx)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                  num_workers=num_workers, pin_memory=pin_memory)
    else:
        valid_loader = None
    test_dataset = MELDDataset(features_path, test_idx)
    all_dataset = MELDDataset(features_path, all_idx)

    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
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
