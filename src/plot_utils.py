'''
Utility functions for visualization
'''

from pathlib import Path
import random
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
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
    
    
def plot_resolution_distribution(all_width, all_height, save_path=None):
    # Set up
    plt.figure(figsize=(8, 6))
    
    # Plotting
    square_res_x = square_res_y = np.arange(250, step=1)
    plt.plot(square_res_x, square_res_y, "--r", label="Square Frame Ratio (1:1)")
    plt.scatter(all_width, all_height, alpha=0.5)
    plt.title("Resolution Distribution")
    plt.xlabel("Width (pixel)")
    plt.ylabel("Height (pixel)")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)

    plt.show()
    
    
def plot_frame_count_distribution(all_frame_count, save_path=None):
    # Set up
    plt.figure(figsize=(8, 6))
    frame_count_dist = all_frame_count.value_counts()
    
    # Plotting
    ax = sns.barplot(
        x=frame_count_dist.index,
        y=frame_count_dist.values,
        color="skyblue",
        edgecolor="black",
        linewidth=1.2
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--")
    
    plt.title("Frame Count Distribution")
    plt.xlabel("Number of Frames")
    plt.ylabel("Number of Videos")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    
def plot_class_balance(labels, save_path=None):
    # Set up
    plt.figure(figsize=(18, 6))
    class_count = labels.value_counts()
    
    # Plotting
    ax = sns.barplot(
        x=class_count.index,
        y=class_count.values,
        hue=class_count.index,
        palette="flare"
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--")
    plt.xticks(rotation=90, fontsize=10)
    
    plt.title("Number of Videos per Class", fontsize=16)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Number of Videos", fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()
    
    
def plot_confusion_matrix(
    y_true, y_pred, labels, display_labels, top_k=10, figsize=(20, 26),
    normalize="true", save_path=None
):
    '''
    Plot confusion matrix and a table of top_k misclassified pairs
    
    Args:
        y_true (lst)        : true labels
        y_pred (lst)        : predictions from model
        labels (lst)        : list of integer labels
        display_labels (lst): labels to display
        top_k (int)         : number of classes with highest confusion to include
                              in confusion matrix, default: 10
        figsize (tuple)     : figure size, default: (20, 26)
        fontsize (float)    : size of texts for labels
        normalize (str)     : option to normalize confusion matrix
                              (same in sklearn.metrics.confusion_matrix),
                              but only accepts 2 value: "true" (normalize
                              by row) and None (no normalization), default: "true"
        save_path (str)     : path to save the plot if provided, default: None

    Returns:
        None
    '''
    
    # Full confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    # Find (i, j) indices (i != j) that have highest confusion
    confusions = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                confusions.append((i, j, cm[i][j]))

    # Sorting and find top-k confused pairs
    top_confusions = sorted(confusions, key=lambda x: x[2], reverse=True)[:top_k]

    # Set up plots
    fig, axes = plt.subplots(
        nrows=2, figsize=figsize, gridspec_kw={"height_ratios": [4, 1]}
    )

    # Plot confusion matrix
    sns.heatmap(
        cm, cmap="Blues", linewidths=0.5, linecolor="gray",
        xticklabels=display_labels, yticklabels=display_labels,
        cbar_kws={"label": "Proportion" if normalize else "Count"}, ax=axes[0]
    )

    axes[0].set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    axes[0].set_xlabel("Predicted Label", fontsize=14)
    axes[0].set_ylabel("True Label", fontsize=14)
    axes[0].tick_params(axis="x", labelsize=13)
    axes[0].tick_params(axis="y", labelsize=13)
    plt.setp(axes[0].get_xticklabels(), rotation=90, ha="right")

    # Table of top_k misclassified pairs
    columns = ["Ground Truth", "Predicted", "Proportion" if normalize else "Count"]
    data = [
        [display_labels[i], display_labels[j], f"{v:.2f}" if normalize else int(v)]
        for i, j, v in top_confusions
    ]

    axes[1].axis("off")
    table = axes[1].table(
        cellText=data, colLabels=columns, loc="center", cellLoc="center",
        colColours=["#d3d3d3"] * len(columns), bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    axes[1].set_title(
        f"Top-{top_k} Misclassified Pairs", fontsize=14, fontweight="bold", pad=10
    )
    
    plt.tight_layout(h_pad=5)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    
    
def plot_training_progress(
    avg_training_losses,
    avg_val_losses,
    precision_scores,
    recall_scores,
    f1_scores,
    lr_changes,
    save_path=None
):
    '''
    Plot training process over epochs, specifically, 3 subplots are created:
    - One plot for average train and validation loss
    - One plot for accuracy and weighted F1 score on validation data
    - One plot for learning rates

    Args:
        avg_training_losses (lst): average training loss
        avg_val_losses (lst)     : average validation loss
        precision_scores (lst)   : precision on validation data
        recall_scores (lst)      : recall on validation data
        f1_scores (lst)          : macro F1 score on validation data
        lr_changes (lst)         : learning rates
        save_path (str)          : path to save the plot if provided, default: None

    Returns:
        None
    '''

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 15))
    n_epochs = [i+1 for i in range(len(avg_training_losses))]

    # Avg Training vs Validation loss
    axes[0].plot(n_epochs, avg_training_losses, label="Train", color="blue")
    axes[0].plot(n_epochs, avg_val_losses, label="Validation", color="red")
    axes[0].set(
        xlabel="Epoch",
        ylabel="Average Loss",
        title="Average Training vs Validation Loss"
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    # Precision, Recall and Macro F1 score on validation data
    axes[1].plot(n_epochs, precision_scores, label="Precision", color="blue")
    axes[1].plot(n_epochs, recall_scores, label="Recall", color="green")
    axes[1].plot(n_epochs, f1_scores, label="Macro F1 Score", color="red")
    axes[1].set(
        xlabel="Epoch",
        ylabel="Score (%)",
        title="Validation Precision, Recall and Macro F1 Score"
    )
    axes[1].legend(loc="lower right")
    axes[1].grid(True)

    # Learning rate
    axes[2].plot(n_epochs, lr_changes)
    axes[2].set(
        xlabel="Epoch",
        ylabel="Learning Rate",
        title="Learning Rate Changes"
    )
    axes[2].grid(True)

    plt.suptitle("Training Process", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()