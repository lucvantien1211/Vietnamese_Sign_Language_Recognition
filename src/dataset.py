'''
Custom dataset class definition for Vietnamese sign language data
'''
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json

from src.data_utils import read_video, nfc_normalize


def collate_fn(batch):
    frames = torch.stack([item["frames"] for item in batch])
    output = {"frames": frames}
    
    if "label" in batch[0] and batch[0]["label"] is not None:
        output["labels"] = torch.tensor([item["label"] for item in batch])

    if "path" in batch[0]:
        output["paths"] = [item["path"] for item in batch]

    return output


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
        ] if mode != "test" else [None] * len(paths)
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        video_path = self.paths[idx]
        label = self.labels[idx]
        frames = read_video(video_path)
        
        if self.transform is not None:
            frames = self.transform(frames)
        
        frames = self._resample_frames(frames)
        frames = self._normalize(frames)
        
        output = {"frames": frames, "label": label} if self.mode != "test" \
            else {"frames": frames, "path": video_path}

        return output
    
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
    
    
class VideoAugmentation:
    '''
    Custom class for video data augmentation. These transformations are
    consistent across all frames for one video
    '''
    
    def __init__(
        self, mode,
        output_size=(224, 224),
        crop_scale=(0.85, 1.0),
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        speed_range=(0.9, 1.1)
    ):
        assert mode in ["train", "validation", "test"], "Invalid value for augmentation mode"
        self.mode = mode
        self.output_size = output_size
        
        if self.mode == "train":
            self.crop_scale = crop_scale
            self.brightness = brightness
            self.contrast = contrast
            self.saturation = saturation
            self.speed_range = speed_range
    
    def __call__(self, frames):
        if self.mode == "train":
            # Speed Augmentation
            frames = self._speed_augment(frames)

            # Random Resized Crop
            frames = self._random_resized_crop(frames)

            # Color Jitter
            frames = self._color_jitter(frames)
            
        else:
            # Only resize for validation and test data
            frames = self._resize(frames)
            
        return frames
    
    def _speed_augment(self, frames):
        '''Changing video speed by resampling frames'''
        T = frames.shape[0]
        speed = random.uniform(self.speed_range[0], self.speed_range[1])

        new_T = int(T / speed)
        if new_T < 4:
            new_T = 4
        if new_T == T:
            return frames

        # Resample frames
        indices = torch.linspace(0, T - 1, new_T).long()
        indices = torch.clamp(indices, 0, T - 1)
        frames = frames[indices]

        return frames
    
    def _resize(self, frames):
        H, W = frames.shape[1], frames.shape[2]
        output_H, output_W = self.output_size
        
        if H != output_H or W != output_W:
            frames = frames.permute(0, 3, 1, 2).float()
            frames = F.interpolate(frames, size=self.output_size, mode='bilinear', align_corners=False)
            frames = frames.permute(0, 2, 3, 1).to(torch.uint8)
            
        return frames

    def _random_resized_crop(self, frames):
        '''Random crop then resize to the desire output size'''
        T, H, W, C = frames.shape

        # Random scale and position
        scale = random.uniform(self.crop_scale[0], self.crop_scale[1])
        crop_h, crop_w = int(H * scale), int(W * scale)

        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        # Crop all frames
        frames = frames[:, top:top+crop_h, left:left+crop_w, :]

        # Resize
        # (T, H, W, C) -> (T, C, H, W) for interpolate
        frames = frames.permute(0, 3, 1, 2).float()
        frames = F.interpolate(frames, size=self.output_size, mode='bilinear', align_corners=False)
        # (T, C, H, W) -> (T, H, W, C)
        frames = frames.permute(0, 2, 3, 1)

        return frames.to(torch.uint8)

    def _color_jitter(self, frames):
        '''Color jitter all frames'''
        # Random parameters (same for all frames)
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        saturation_factor = 1.0 + random.uniform(-self.saturation, self.saturation)

        frames = frames.float()

        # Brightness
        frames = frames * brightness_factor

        # Contrast
        mean = frames.mean(dim=(1, 2), keepdim=True)
        frames = (frames - mean) * contrast_factor + mean

        # Saturation
        gray = frames.mean(dim=-1, keepdim=True)
        frames = gray + (frames - gray) * saturation_factor

        # Clamp to valid range
        frames = torch.clamp(frames, 0, 255)

        return frames.to(torch.uint8)