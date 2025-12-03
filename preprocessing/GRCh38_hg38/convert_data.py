"""
Script to convert arrow data to jsonl 
to build infini-gram index
"""

import os
import json
import argparse
from datasets import load_dataset, Dataset, DatasetDict

def iter_dataset_from_file(arrow_file):
    """
    return dataset from given arrow file
    """
    print(f"loading dataset from: {arrow_file}")
    ds_obj = load_dataset("arrow", data_files=arrow_file)
    
    if isinstance(ds_obj, DatasetDict):
        for split_name, ds in ds_obj.items():
            print(f"found split '{split_name}' with {len(ds)} examples")
            for ex in ds:
                yield ex
    elif isinstance(ds_obj, Dataset):
        print(f"found single split with {len(ds_obj)} examples")
        for ex in ds_obj:
            yield ex
    else:
        raise TypeError(f"Unsupported dataset type loaded from {arrow_file}: {type(ds_obj)}")


def main():
    """
    Given arrow dataset path, output dir, 
    convert arrow to jsonl into one file with given name
    """
    parser = argparse.ArgumentParser(
        description="convert one or more arrow dataset to a single jsonl"
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="one or more paths to arrow files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="output directory for the merged JSONL file",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="data.jsonl",
        help="output jsonl filename (default: data.jsonl)",
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)
    
    print("converting the following arrow files:")
    for i in args.files:
        print(f"File: {i}")
    print(f"\nwriting merged jsonl to: {out_path}")
    
    num = 0
    with open(out_path, "w") as f:
        for i in args.files:
            for example in iter_dataset_from_file(i):
                seq = example["sequence"]
                obj = {"text": seq}
                f.write(json.dumps(obj) + "\n")
                num += 1
                
                if num % 10000 == 0:
                    print(f"  Wrote {num} examples so far...")
    
    print(f"num of sequences written: {num}")
    print(f"Output JSONL: {out_path}")


if __name__ == "__main__":
    print("In main...")
    main()
    print("Done.")
