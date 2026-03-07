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
    n_sample_classes = len(sample_classes)
    fig, axes = plt.subplots(n_sample_classes, n_frames, figsize=(15,8))
    
    for row, cls in enumerate(sample_classes):
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
        for col in range(n_frames):
            ax = axes[row, col]
            ax.imshow(sample_frames[col])
            ax.axis("off")
            
        # Labeling each row
        fig.text(
            0.1,
            1 - (row + 0.5) / n_sample_classes,
            cls,
            ha="left",
            va="center",
            fontsize=16
        )
                
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.suptitle("Sample Frames", fontsize=16)
    plt.subplots_adjust(top=0.88, left=0.2)

    if save_path:
        plt.savefig(save_path)

    plt.show()