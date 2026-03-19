'''
Utility script for converting label mapping (label -> integer) in the
file 'label_mapping.pkl' to json format.
Reason: some platform like Hugging Face mark pickle file as dangerous,
so converting the mapping to JSON format is safer and more portable
'''

from pathlib import Path
import pickle
import json
import argparse

from data_utils import nfc_normalize

ROOT = Path(__file__).parents[1] / "data/dataset"
DEFAULT_PKL = ROOT / "label_mapping.pkl"
DEFAULT_JSON = ROOT / "label_mapping.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert label_mapping.pkl to .json")

    parser.add_argument(
        "--pkl_path",
        type=str,
        default=DEFAULT_PKL,
        help="Path to the original label_mapping.pkl"
    )

    parser.add_argument(
        "--json_path",
        type=str,
        default=DEFAULT_JSON,
        help="Path to output JSON file"
    )

    return parser.parse_args()


def convert_pkl_to_json(pkl_path, json_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        
    data = {nfc_normalize(k): v for k, v in data.items()}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Saved json to {json_path}")
    
    
def main():
    args = parse_args()
    
    convert_pkl_to_json(pkl_path=args.pkl_path, json_path=args.json_path)
    
    
if __name__ == "__main__":
    main()