import argparse
import logging
from datetime import datetime
from pathlib import Path
import time
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import models

from src.train_utils import *
from src.plot_utils import plot_training_progress
from src.dataset import *
from src.models import *


ROOT = Path(__file__).parents[1]
DEFAULT_TRAIN_ROOT = ROOT / "data/dataset/train"
DEFAULT_LABEL_MAPPING_PATH = ROOT / "data/dataset/label_mapping.json"
DEFAULT_METADATA_PATH = ROOT / "data/dataset/video_metadata.csv"
DEFAULT_LOG_DIR = ROOT / "logs"
DEFAULT_MODEL_SAVE_DIR = ROOT / "models"
DEFAULT_TRAIN_PROGRESS_DIR = ROOT / "train_progress"
DEFAULT_VALIDATION_RESULTS_DIR = ROOT / "validation_results"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_root", type=str, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--label_mapping_path", type=str, default=DEFAULT_LABEL_MAPPING_PATH)
    parser.add_argument("--metadata_path", type=str, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_SAVE_DIR)
    parser.add_argument("--train_progress_dir", type=str, default=DEFAULT_TRAIN_PROGRESS_DIR)
    parser.add_argument("--validation_results_dir", type=str, default=DEFAULT_VALIDATION_RESULTS_DIR)

    parser.add_argument("--model", type=str, default="crnn")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    logger, log_file = setup_logger(log_dir=args.log_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("========== TRAINING START ==========")
    logger.info(f"Log file: {log_file}")

    # log config
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    logger.info(f"device: {device}")
    
    logger.info("====================================")
    
    # ==== Split data ====
    train_paths, val_paths = split_train_val_paths(
        args.train_root,
        args.metadata_path,
        args.seed
    )

    logger.info(f"Train samples: {len(train_paths)}")
    logger.info(f"Val samples: {len(val_paths)}")
    
    # ==== Dataset & Loader ====
    ## Augmentation
    train_transforms = VideoAugmentation(mode="train")
    val_transforms = VideoAugmentation(mode="validation")
    
    train_dataset = VSLDataset(
        paths=train_paths,
        label_mapping_path=args.label_mapping_path,
        mode="train",
        transform=train_transforms,
        target_frames=16
    )
    
    val_dataset = VSLDataset(
        paths=val_paths,
        label_mapping_path=args.label_mapping_path,
        mode="validation",
        transform=val_transforms,
        target_frames=16
    )

    ## Balance sampler for train dataset
    balanced_sampler = create_balanced_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        sampler=balanced_sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # ==== Model ====
    if args.model == "crnn":
        model = CRNN(
            num_classes=len(train_dataset.label2id),
            resnet_pretrained_weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
    elif args.model == "convnext-transformer":
        model = ConvNeXtTransformer(
            num_classes=len(train_dataset.label2id),
            convnext_pretrained_weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        model.freeze_convnext_features(freeze_until=3)
    else:
        logger.info(f"The model {args.model} is not supported. Ending training ...")
        return
    
    # Set up paths
    model_save_dir = Path(args.model_dir) / args.model
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    train_progress_dir = Path(args.train_progress_dir)
    train_progress_dir.mkdir(exist_ok=True)
    
    validation_results_dir = Path(args.validation_results_dir)
    validation_results_dir.mkdir(exist_ok=True)
    
    log_datetime = re.search(r'train_(\d{8}_\d{6})', log_file.stem).group(1)
    model_save_path = model_save_dir / f"best_model_{log_datetime}.safetensors"
    train_progress_path = train_progress_dir / f"train_progress_{log_datetime}.png"
    validation_results_path = validation_results_dir / f"validation_results_{log_datetime}.png"
    
    # ==== Training ====
    train_losses, val_losses, precision_scores,\
        recall_scores, f1_scores, learning_rates = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=device,
            save_path=model_save_path,
            validation_cm_path=validation_results_path
        )
        
    # Plot train progress
    plot_training_progress(
        train_losses,
        val_losses,
        precision_scores,
        recall_scores,
        f1_scores,
        learning_rates,
        save_path=train_progress_path
    )
    logger.info(f"Training Progress Plot is saved at: {train_progress_path}")
    
    
if __name__ == "__main__":
    main()