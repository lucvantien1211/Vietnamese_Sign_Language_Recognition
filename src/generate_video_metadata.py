'''
Utility script for creating a csv file containing
metadata for video in the train dataset
'''


import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from data_utils import get_all_path, get_video_metadata


ROOT = Path("./data/dataset")
TRAIN_DIR = ROOT / "train"


def main(train_dir=TRAIN_DIR, output_dir=ROOT):
    data = []
    
    all_path = get_all_path(train_dir)
    n_videos = len(all_path)
    output_path = output_dir / "video_metadata.csv"
    
    for idx in tqdm(range(n_videos), desc="Generating video metadata"):
        height, width, frame_count, fps = get_video_metadata(all_path[idx])

        data.append({
            "video_path": all_path[idx].relative_to(train_dir),
            "height": height,
            "width": width,
            "frame_count": frame_count,
            "fps": fps
        })
        
        if (idx+1) % 100 == 0:
            print(f"Complete extracting metadata for {idx+1}/{n_videos}")
            
    print("Metadata extraction completed, generating csv file")
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print("Completed successfully")
    
    
if __name__ == "__main__":
    main()