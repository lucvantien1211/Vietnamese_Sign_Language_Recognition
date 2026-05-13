'''
Custom dataset class definition for Vietnamese sign language data

Credit to AI VIET NAM: https://aivietnam.edu.vn for custom Dataset class
and VideoAugmentation class
'''
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json

from src.data_utils import read_video, nfc_normalize


def collate_fn(batch):
    '''
    Custom collate function for VSLDataset
    '''
    
    frames = torch.stack([item["frames"] for item in batch])
    output = {"frames": frames}
    
    if "label" in batch[0] and batch[0]["label"] is not None:
        output["labels"] = torch.tensor([item["label"] for item in batch])

    if "path" in batch[0]:
        output["paths"] = [item["path"] for item in batch]

    return output


class VSLDataset(Dataset):
    '''
    Custom dataset class for Vietnamese Sign Language video data

    Args:
        paths (lst | Path)              : list of video file paths
        label_mapping_path (str | Path) : path to the JSON file containing
                                          label-to-id mapping
        mode (str)                      : dataset mode, must be one of
                                          ["train", "validation", "test"],
                                          default: "train"
        transform (callable)            : transformation pipeline applied
                                          to video frames, default: None
        norm_stats (dict)               : normalization statistics containing
                                          "mean" and "std", default:
                                          {
                                              "mean": [0.485, 0.456, 0.406],
                                              "std": [0.229, 0.224, 0.225]
                                          }
        target_frames (int)             : target number of frames after
                                          resampling, default: 32

    Attributes:
        paths (lst)             : list of video file paths
        mode (str)              : dataset mode
        transform (callable)    : transformation pipeline
        norm_stats (dict)       : normalization statistics
        target_frames (int)     : target number of frames
        label2id (dict)         : mapping from class label to integer id
        labels (lst)            : list of labels corresponding to
                                  each video path

    Returns:
        dict: for train/validation mode:
              {
                  "frames": normalized video tensor,
                  "label" : class label id
              }

              for test mode:
              {
                  "frames": normalized video tensor,
                  "path"  : video path
              }
    '''
    
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
        '''
        Resample video frames to a fixed number of frames

        Args:
            frames (torch.Tensor): input video frames with shape
                                   (num_frames, height, width, channels)

        Returns:
            torch.Tensor: resampled video frames with exactly
                          self.target_frames frames

        Notes:
            - If the video contains more frames than target_frames,
            frames are sampled evenly across the sequence
            - If the video contains fewer frames than target_frames,
            the last frame is repeated for padding
        '''
        
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
        '''
        Normalize video frames using channel-wise mean and standard deviation

        Args:
            frames (torch.Tensor): input video frames with shape
                                   (num_frames, height, width, channels)

        Returns:
            torch.Tensor: normalized video tensor with shape
                          (num_frames, channels, height, width)
        '''
        
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor(self.norm_stats["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(self.norm_stats["std"]).view(1, 3, 1, 1)
        return (frames - mean) / std
    
    
class VideoAugmentation:
    '''
    Custom video augmentation pipeline with temporally consistent
    transformations across all frames in a video

    Args:
        mode (str)                : augmentation mode, must be one of
                                    ["train", "validation", "test"]
        output_size (tuple)       : target output resolution
                                    (height, width), default: (224, 224)
        crop_scale (tuple)        : range of random crop scale factors,
                                    default: (0.85, 1.0)
        brightness (float)        : maximum brightness adjustment factor,
                                    default: 0.2
        contrast (float)          : maximum contrast adjustment factor,
                                    default: 0.2
        saturation (float)        : maximum saturation adjustment factor,
                                    default: 0.2
        speed_range (tuple)       : range of video speed scaling factors,
                                    default: (0.9, 1.1)

    Attributes:
        mode (str)                : augmentation mode
        output_size (tuple)       : target output resolution
        crop_scale (tuple)        : crop scale range for random crop
        brightness (float)        : brightness jitter strength
        contrast (float)          : contrast jitter strength
        saturation (float)        : saturation jitter strength
        speed_range (tuple)       : video speed augmentation range

    Notes:
        - All augmentations are applied consistently across all frames
          of a single video
        - Validation and test modes only apply resizing
        - Training mode applies speed augmentation, random resized crop,
          and color jitter
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
        '''
        Apply augmentation pipeline to video frames

        Args:
            frames (torch.Tensor): input video frames with shape
                                   (num_frames, height, width, channels)

        Returns:
            torch.Tensor: augmented video frames
        '''
        
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
        '''
        Apply temporal speed augmentation by resampling frames

        Args:
            frames (torch.Tensor): input video frames with shape
                                   (num_frames, height, width, channels)

        Returns:
            torch.Tensor: temporally resampled video frames

        Notes:
            - A random speed factor is sampled from self.speed_range
            - Frames are evenly resampled to simulate faster or slower motion
            - Minimum output length is 4 frames
        '''
        
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
        '''
        Resize all video frames to the target output size

        Args:
            frames (torch.Tensor): input video frames with shape
                                   (num_frames, height, width, channels)

        Returns:
            torch.Tensor: resized video frames with shape
                          (num_frames, output_height, output_width, channels)
        '''
        
        H, W = frames.shape[1], frames.shape[2]
        output_H, output_W = self.output_size
        
        if H != output_H or W != output_W:
            frames = frames.permute(0, 3, 1, 2).float()
            frames = F.interpolate(frames, size=self.output_size, mode='bilinear', align_corners=False)
            frames = frames.permute(0, 2, 3, 1).to(torch.uint8)
            
        return frames

    def _random_resized_crop(self, frames):
        '''
        Apply random resized crop consistently across all frames

        Args:
            frames (torch.Tensor): input video frames with shape
                                   (num_frames, height, width, channels)

        Returns:
            torch.Tensor: cropped and resized video frames

        Notes:
            - A random crop region is sampled once and applied to all frames
            - Cropped frames are resized to self.output_size
        '''
        
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
        '''
        Apply color jitter augmentation consistently across all frames

        Args:
            frames (torch.Tensor): input video frames with shape
                                   (num_frames, height, width, channels)

        Returns:
            torch.Tensor: color-jittered video frames

        Notes:
            - Brightness, contrast, and saturation factors are sampled once
            and applied consistently to all frames
            - Pixel values are clamped to the valid range [0, 255]
        '''
        
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