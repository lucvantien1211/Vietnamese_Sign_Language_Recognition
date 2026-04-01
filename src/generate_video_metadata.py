'''
Utility script for creating a csv file containing
metadata for video in the train dataset
'''


import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import argparse

from src.data_utils import get_all_path, get_video_metadata


DEFAULT_ROOT = Path(__file__).parents[1] / "data/dataset"
DEFAULT_OUTPUT = DEFAULT_ROOT / "video_metadata.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video metadata CSV")

    parser.add_argument(
        "--root_dir",
        type=str,
        default=DEFAULT_ROOT,
        help="Path to dataset root directory"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to output CSV file"
    )

    return parser.parse_args()


def extract_metadata(root, output_path):
    data = []
    
    root = Path(root)
    train_dir = root / "train"
    output_path = Path(output_path)
    
    all_path = get_all_path(train_dir)
    n_videos = len(all_path)
    
    for idx in tqdm(range(n_videos), desc="Generating video metadata"):
        height, width, frame_count, fps = get_video_metadata(all_path[idx])

        data.append({
            "video_name": all_path[idx].name,
            "label": all_path[idx].parent.name,
            "height": height,
            "width": width,
            "frame_count": frame_count,
            "fps": fps
        })
        
        if (idx+1) % 100 == 0:
            print(f"\nCompleted {idx+1}/{n_videos}")
            
    print("Metadata extraction completed. Saving CSV ...")
    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("Completed successfully")
    

def main():
    args = parse_args()
    
    extract_metadata(
        root=args.root_dir,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()