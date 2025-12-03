"""
Script to reverse sequences in jsonl data
to build reversed infini-gram index

to run:
    $ module restore deps

    $ python reverse_converted_data.py \
    --input artifacts/datasets/human_reference_genome/6kbp/jsonl/train/human_reference_genome-train.jsonl \
    --output artifacts/datasets/human_reference_genome/6kbp/jsonl/train_reversed/human_reference_genome_reversed-train.jsonl

    $ python reverse_converted_data.py \
    --input artifacts/datasets/human_reference_genome/6kbp/jsonl/train-validate/human_reference_genome-train-validation.jsonl \
    --output artifacts/datasets/human_reference_genome/6kbp/jsonl/train-validate_reversed/human_reference_genome_reversed-train-validation.jsonl
"""

import os
import json
import argparse

def reverse_sequence(seq):
    return seq[::-1]

def main():
    """
    Given jsonl file, reverse each sequence, 
    and write to new file
    """
    parser = argparse.ArgumentParser(
        description="reverse sequences in jsonl file"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to input jsonl file with forward sequences",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path to output jsonl file with reversed sequences",
    )
    
    args = parser.parse_args()
        
    print(f"Reading from: {args.input}")
    print(f"Writing to: {args.output}")
    
    num_processed = 0
    num_total_chars = 0
    
    with open(args.input, "r") as infile, open(args.output, "w") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                print(f"spiked {num_processed} entry")
                continue
            
            obj = json.loads(line)
            seq = obj["text"]
            
            reversed_seq = reverse_sequence(seq)
            reversed_obj = {"text": reversed_seq}
            outfile.write(json.dumps(reversed_obj) + "\n")
            
            num_processed += 1
            num_total_chars += len(seq)
            
            if num_processed % 10000 == 0:
                print(f"  Processed {num_processed} sequences...")
                        
    print(f"\nProcessing complete!")
    print(f"Total sequences processed: {num_processed}")
    print(f"Total characters reversed: {num_total_chars}")
    print(f"Average sequence length: {num_total_chars / num_processed:.2f}" if num_processed > 0 else "N/A")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    print("Starting sequence reversal...")
    main()
    print("Done.")