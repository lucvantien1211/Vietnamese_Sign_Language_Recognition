'''
Custom dataset class definition for Vietnamese sign language data
'''

import torch
from torch.utils.data import Dataset
import json

from src.data_utils import read_video, nfc_normalize


class VSLDataset(Dataset):
    def __init__(
        self, paths, label_mapping_path,
        mode="train", transform=None,
        norm_stats={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }, target_frames=32
    ):
        assert mode in ["train", "validation", "test"], "Invalid value for dataset mode"
        super().__init__()
        self.paths = paths
        self.mode = mode
        self.transform = transform
        self.norm_stats = norm_stats
        self.target_frames = target_frames
        
        with open(label_mapping_path, "r", encoding="utf-8") as f:
            self.label2id = json.load(f)
            
        self.labels = [
            self.label2id[nfc_normalize(video_path.parent.name)]
            for video_path in paths
        ]
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        video_path = self.paths[idx]
        label = self.labels[idx]
        frames = read_video(video_path)
        frames = self._resample_frames(frames)
        frames = self._normalize(frames)
        return {"frames": frames, "label": label}
    
    def _resample_frames(self, frames):
        total = frames.shape[0]
        if total >= self.target_frames:
            indices = torch.linspace(0, total - 1, self.target_frames).long()
        else:
            indices = torch.arange(total)
            pad = self.target_frames - total
            indices = torch.cat([indices, indices[-1].repeat(pad)])

        frames = frames[indices]

        return frames
        
    def _normalize(self, frames):
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor(self.norm_stats["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(self.norm_stats["std"]).view(1, 3, 1, 1)
        return (frames - mean) / std