'''
Utility functions for handling data
'''

import torch
import cv2
import numpy as np
import unicodedata
from pathlib import Path


def get_sample_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")
    frames = torch.from_numpy(np.stack(frames, axis=0))
    return frames


def get_all_path(root, labeled=True):
    root = Path(root)
    all_path = []
    
    if labeled:
        for cls in root.iterdir():
            if not cls.is_dir():
                continue
            
            for video_path in cls.iterdir():
                if not video_path.is_file():
                    continue
                
                all_path.append(video_path)
                
    else:
        for video_path in root.iterdir():
            if not video_path.is_file():
                continue
            
            all_path.append(video_path)
            
    return all_path


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    
    return height, width, frame_count, fps


def nfc_normalize(s):
    return unicodedata.normalize("NFC", s)