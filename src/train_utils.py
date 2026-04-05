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
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.plot_utils import plot_confusion_matrix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_train_val_paths(train_root, metadata_path, random_state=None):
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


def validate(model, dataloader, criterion, device):
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
    save_path="best_model.pth",
    validation_cm_path="validation_cm.png"
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=3
    )

    train_losses = []
    val_losses = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    learning_rates = []

    best_f1 = 0.0
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
        
        scheduler.step(val_loss)
        
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
            torch.save(model.state_dict(), save_path)

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
            
    total_time = time.time() - start_time
    
    logger.info("========== TRAINING END ==========")
    logger.info(f"Best F1: {best_f1:.2f}%")
    logger.info(f"Total Time: {total_time/60:.2f} minutes")
            
    return (
        train_losses, val_losses, precision_scores,
        recall_scores, f1_scores, learning_rates
    )
    
    
def setup_logger(log_dir="logs"):
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
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)