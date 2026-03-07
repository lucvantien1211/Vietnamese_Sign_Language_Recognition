'''
Utility functions for visualization
'''

from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_utils import get_sample_frames


# Function for plotting sample frames from videos in training set
def plot_sample_frames(root, sample_classes=["Ăn", "Nghỉ ngơi", "Chạy"], n_frames=5, save_path=None):
    
    # Set up
    plt.figure(figsize=(12, 6))
    n_sample_classes = len(sample_classes)
    
    for y, cls in enumerate(sample_classes):
        # Get sample video
        sample_dir = Path(root) / cls
        if not (sample_dir.exists() and sample_dir.is_dir()):
            print(f"The directory \"{cls}\" is not available, skipping")
            continue
        
        all_path = [video_path for video_path in sample_dir.iterdir()]
        sample_path = random.choice(all_path)
        
        # Get sample frames
        attempt = 0
        sample_frames = get_sample_frames(sample_path, num_frames=n_frames)
        if sample_frames is None:
            while sample_frames is None and attempt < 10:
                sample_path = random.choice(all_path)
                sample_frames = get_sample_frames(sample_path, num_frames=n_frames)
                attempt += 1
        
        # Plotting
        for idx, frame in enumerate(sample_frames):
            plt_idx = n_frames * y + idx + 1
            plt.subplot(n_sample_classes, n_frames, plt_idx)
            plt.imshow(frame)
            plt.axis("off")
            if idx == (n_frames // 2):
                plt.title(cls)
                
    plt.tight_layout()
    plt.suptitle("Sample Frames", fontsize=16)
    plt.subplots_adjust(top=0.88)

    if save_path:
        plt.savefig(save_path)

    plt.show()