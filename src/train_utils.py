'''
Utility functions for training model

Credit to AI VIET NAM: https://aivietnam.edu.vn for some of the functions
'''

from pathlib import Path
import logging
from datetime import datetime
import time

import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm.auto import tqdm

from safetensors.torch import save_file

from src.plot_utils import plot_confusion_matrix


def set_seed(seed):
    '''
    Set random seeds for reproducibility

    Args:
        seed (int): random seed value

    Returns:
        None

    Notes:
        - Sets seeds for Python random module
        - Sets seeds for NumPy random generator
        - Sets seeds for PyTorch CPU and CUDA operations
    '''
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_train_val_paths(train_root, metadata_path, random_state=None):
    '''
    Split dataset paths into training and validation sets

    Args:
        train_root (str | Path)   : root directory containing
                                    training videos organized by class
        metadata_path (str | Path): path to metadata CSV file
                                    containing video names and labels
        random_state (int)        : random seed for reproducible splitting,
                                    default: None

    Returns:
        tuple:
            - train_paths (lst): list of training video paths
            - val_paths (lst)  : list of validation video paths

    Notes:
        - Uses stratified splitting to preserve class distribution
        - Validation set size is fixed at 20% of the dataset
    '''
    
    train_root = Path(train_root)
    metadata_df = pd.read_csv(metadata_path)
    
    X = metadata_df[["label", "video_name"]]
    y = metadata_df["label"]
    
    X_train, X_val, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    
    train_paths = (train_root / X_train["label"] / X_train["video_name"]).to_list()
    val_paths = (train_root / X_val["label"] / X_val["video_name"]).to_list()
    
    return train_paths, val_paths


def create_balanced_sampler(dataset):
    '''
    Create a weighted random sampler for imbalanced datasets
    
    Args:
        dataset (VSLDataset): dataset object containing a `labels`
                              attribute with class labels

    Returns:
        WeightedRandomSampler: sampler that balances class sampling
                               probability during training

    Notes:
        - Sample weights are computed inversely proportional to
          class frequency
        - Minority classes receive higher sampling probability
        - Sampling is performed with replacement
    '''
    
    all_labels = dataset.labels

    class_counts = np.bincount(all_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in all_labels]
    sample_weights = torch.FloatTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def validate(model, dataloader, criterion, device):
    '''
    Evaluate a model on a validation dataset

    Args:
        model (nn.Module)         : model to evaluate
        dataloader (DataLoader)   : validation dataloader
        criterion (nn.Module)     : loss function
        device (torch.device)     : computation device

    Returns:
        tuple:
            - avg_loss (float): average validation loss
            - metrics (dict)  : dictionary containing:
                                {
                                    "precision": macro precision (%),
                                    "recall"   : macro recall (%),
                                    "f1"       : macro F1-score (%)
                                }
            - preds (lst)     : predicted class indices
            - labels_all (lst): ground-truth class indices

    Notes:
        - Model is evaluated in inference mode using model.eval()
        - Gradient computation is disabled with torch.no_grad()
        - Metrics are computed using macro averaging
    '''
    
    model.eval()
    total_loss, preds, labels_all = 0, [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            frames, labels = batch["frames"].to(device), batch["labels"].to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            preds.extend(predicted.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_all, preds, average="macro", zero_division=0
    )
    
    return (
        total_loss / len(dataloader),
        {"precision": precision*100, "recall": recall*100, "f1": f1*100},
        preds,
        labels_all
    )
    
    
def train_epoch(model, dataloader, criterion, optimizer, device):
    '''
    Train the model for one epoch

    Args:
        model (nn.Module)         : model to train
        dataloader (DataLoader)   : training dataloader
        criterion (nn.Module)     : loss function
        optimizer (Optimizer)     : optimization algorithm
        device (torch.device)     : computation device

    Returns:
        tuple:
            - avg_loss (float): average training loss
            - lr (float)      : learning rate used during the epoch

    Notes:
        - Model is set to training mode using model.train()
        - Gradients are cleared before each optimization step
        - Parameters are updated using backpropagation
    '''
    
    total_loss = 0
    progress = tqdm(dataloader, desc="Training")
    
    model.train()
    for batch in progress:
        frames, labels = batch["frames"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        lr = optimizer.param_groups[0]["lr"]
        progress.set_postfix({"loss": f"{total_loss / (len(progress)+1e-9):.4f}"})

    return total_loss / len(dataloader), lr


def train_model(
    model, train_loader, val_loader, logger,
    num_epochs=10, lr=5e-4, device="cuda",
    early_stopping_patience=3,
    save_path="best_model.safetensors",
    validation_cm_path="validation_cm.png"
):
    '''
    Train and validate a model with early stopping and checkpoint saving

    Args:
        model (nn.Module)              : model to train
        train_loader (DataLoader)      : training dataloader
        val_loader (DataLoader)        : validation dataloader
        logger (logging.Logger)        : logger for training progress
        num_epochs (int)               : maximum number of training epochs,
                                         default: 10
        lr (float)                     : initial learning rate,
                                         default: 5e-4
        device (str | torch.device)    : computation device,
                                         default: "cuda"
        early_stopping_patience (int)  : number of consecutive epochs
                                         without improvement before
                                         stopping training, default: 3
        save_path (str | Path)         : path to save the best model
                                         weights, default: "best_model.safetensors"
        validation_cm_path (str | Path): path to save the validation
                                         confusion matrix plot,
                                         default: "validation_cm.png"

    Returns:
        tuple:
            - train_losses (lst)    : training loss history
            - val_losses (lst)      : validation loss history
            - precision_scores (lst): validation precision history
            - recall_scores (lst)   : validation recall history
            - f1_scores (lst)       : validation F1-score history
            - learning_rates (lst)  : learning rate history

    Notes:
        - Uses CrossEntropyLoss with label smoothing
        - Optimizer: AdamW
        - Scheduler: CosineAnnealingWarmRestarts
        - Best model is selected based on validation macro F1-score
        - Early stopping is triggered when validation F1 does not improve
          for `early_stopping_patience` consecutive epochs
        - Confusion matrix is generated and saved whenever a new best
          model is found
    '''
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    train_losses = []
    val_losses = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    learning_rates = []

    best_f1 = 0.0
    best_f1_epoch = 1
    early_stopping_cnt = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        logger.info(f"===== Epoch {epoch+1}/{num_epochs} =====")
        
        train_loss, lr = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_metrics, preds, labels_all = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        precision_scores.append(val_metrics["precision"])
        recall_scores.append(val_metrics["recall"])
        f1_scores.append(val_metrics["f1"])
        learning_rates.append(lr)

        logger.info(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Precision: {val_metrics['precision']:.2f}% | "
            f"Val Recall: {val_metrics['recall']:.2f}% | "
            f"Val F1: {val_metrics['f1']:.2f}% | "
            f"LR: {lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if val_metrics["f1"] > best_f1:
            label_mapping = train_loader.dataset.label2id
            best_f1 = val_metrics["f1"]
            best_f1_epoch = epoch + 1
            early_stopping_cnt = 0
            save_file(model.state_dict(), save_path)

            plot_confusion_matrix(
                labels_all, preds,
                labels=[v for k, v in sorted(label_mapping.items(), key=lambda x: x[1])],
                display_labels=[k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])],
                top_k=10,
                figsize=(20, 24),
                normalize="true",
                save_path=validation_cm_path
            )
            
            logger.info(f"✓ Best model saved with F1: {best_f1:.2f}%")
            logger.info(f"✓ Best validation results saved at: {validation_cm_path}")
        
        else:
            early_stopping_cnt += 1
            
        if early_stopping_cnt == early_stopping_patience:
            logger.info(
                f"Early stopping triggered. Best macro F1: {best_f1:.2f}, "
                f"achieved on epoch {best_f1_epoch}"
            )
            break
            
    total_time = time.time() - start_time
    
    logger.info("========== TRAINING END ==========")
    logger.info(f"Best F1: {best_f1:.2f}%")
    logger.info(f"Total Time: {total_time/60:.2f} minutes")
            
    return (
        train_losses, val_losses, precision_scores,
        recall_scores, f1_scores, learning_rates
    )
    
    
def setup_logger(log_dir="logs"):
    '''
    Create and configure a logger for training

    Args:
        log_dir (str | Path): directory to store log files,
                               default: "logs"

    Returns:
        tuple:
            - logger (logging.Logger): configured logger instance
            - log_file (Path)        : path to the generated log file

    Notes:
        - Logs are written to both console and file
        - Log filename includes the current timestamp
        - Log format:
          "YYYY-MM-DD HH:MM:SS | LEVEL | MESSAGE"
    '''
    
    Path(log_dir).mkdir(exist_ok=True)

    log_file = Path(log_dir) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


def seed_worker(worker_id):
    '''
    Initialize random seeds for a DataLoader worker

    Args:
        worker_id (int): worker process ID assigned by DataLoader

    Returns:
        None

    Notes:
        - Ensures reproducibility across DataLoader workers
        - Seeds NumPy and Python random module using
          the worker-specific PyTorch seed
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)