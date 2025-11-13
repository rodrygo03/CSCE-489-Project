"""Script to mask data using the nucleotide transformer tokenizer"""

from transformers import AutoTokenizer
import torch

model_path = "../../nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref/"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

print(model_path)

# # Choose the length to which the input sequences are padded. By default, the 
# # model max length is chosen, but feel free to decrease it as the time taken to 
# # obtain the embeddings increases significantly with it.
max_length = tokenizer.model_max_length

# # Create a dummy dna sequence and tokenize it
sequences = ["ATTCCGATTCCGATTCCG", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]
tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", max_length = max_length)["input_ids"]

print(tokens_ids)

