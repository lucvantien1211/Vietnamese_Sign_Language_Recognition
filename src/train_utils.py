from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


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