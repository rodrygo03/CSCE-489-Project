"""
Script to reproduce accuracy and perplexity for the 
nucleotide-transformer's ability reconstruct human genetic variants
following approach in paper
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from datasets import load_from_disk

import os
import numpy as np
from tqdm import tqdm

def forward_pass(model, masked_seq):
    """
    Run forward pass
    return log probabilities
    """
    model.eval()
    device = next(model.parameters()).device

    masked_seq = masked_seq.to(device).unsqueeze(0)  # (1) -> (1, L)
    with torch.no_grad():
        logits = model(masked_seq).logits

    return F.log_softmax(logits, dim=-1)
    

# A.5.2:
#   The masked sequence is then fed to the model and the probabilities over tokens at
#   each masked position are retrieved. The loss function l(θ,s) and the accuracy acc(θ,s) are defined as
#   follows:

def avg_loss(log_probability, original_seq, positions_masked):
    """
    Compute the avg loss for a masked sequence
    return avg loss - avg negative log-likelihood 
    """
    device = log_probability.device
    original_seq = original_seq.to(device)

    # convert data types to tensors for vectorization
    masked_positions = torch.tensor(positions_masked, dtype=torch.int64, device=device)
    masked_tokens = original_seq[masked_positions].to(device)

    # log likelihood at all masked positions
    log_likelihood = log_probability[0, masked_positions, masked_tokens]

    return -log_likelihood.mean().item() 


def avg_accuracy(log_probability, original_seq, positions_masked):
    """
    Compute the accuracy for a masked sequence 
    return accuracy - fraction of correctly predicted masked tokens in a seq
    """
    device = log_probability.device
    original_seq = original_seq.to(device)

    masked_positions = torch.tensor(positions_masked, dtype=torch.int64, device=device)
    masked_tokens = original_seq[masked_positions].to(device)

    # bool vector with outcome of prediction at masked positions
    pred_tokens = log_probability[0, masked_positions].argmax(dim=-1)
    correct_pred = (pred_tokens == masked_tokens).float()
    
    return correct_pred.mean().item()


#   The perplexity is usually defined in the context of autoregressive generative models. Here, we rely
#   on an alternative definition used in Rives [4], and define it as the exponential of the loss function
#   computed over the masked positions
def avg_perplexity(avg_loss):
    """
    Compute the model's perplexity, the lower the better
    return perplexity - exponential of the models loss averaged per token
    """
    return 2 ** avg_loss


# The Nucleotide Transformer model learned to reconstruct human genetic variants:
#   Median Accuracy: 0.202
#   Perplexity for human reference genome is not reported in paper but rather on another dataset
#   TODO: integrate independent dataset of genetically diverse human genomes & replicate perplexity
def reproduce_nt_metrics(model, schema, npz_name):
    """
    Reproduce InstaDeep's results for recontructing human genetic varaints
    with the 500M human reference model
    """
    # collect sequence-level metrics
    loss = []
    accuracy = []
    perplexity = []
    num_seqs = 0

    for seq in tqdm(schema, desc=f"Collecting Seqs-level metrics for {npz_name}"):
        original_seq = torch.tensor(seq["original_tokens"], dtype=torch.int64)
        masked_seq = torch.tensor(seq["masked_tokens"], dtype=torch.int64)
        positions_masked = seq["positions_masked"]

        if (len(positions_masked) == 0):
            print(f"skiped seq {num_seqs}")
            continue
        
        log_probability = forward_pass(model, masked_seq)
        seq_avg_loss = avg_loss(log_probability, original_seq, positions_masked)
        seq_perplexity = avg_perplexity(seq_avg_loss)
        seq_accuracy = avg_accuracy(log_probability, original_seq, positions_masked)

        loss.append(seq_avg_loss)
        accuracy.append(seq_accuracy)
        perplexity.append(seq_perplexity)
        num_seqs += 1
    
    loss = np.array(loss)
    accuracy = np.array(accuracy)
    perplexity = np.array(perplexity)
    np.savez(
        npz_name,
        loss=loss,
        accuracy=accuracy,
        perplexity=perplexity,
    )

    print(f"Number of Sequences: {num_seqs}")
    print(f"Mean Avg Loss: {loss.mean()}")
    print(f"Mean Accuracy: {accuracy.mean()}")
    print(f"Median Accuracy: {np.median(accuracy)}")
    print(f"Mean Perplexity: {perplexity.mean()}")


if __name__ == "__main__":
    # be sure to run download_nt.sh to download model
    print("Loading Nucleotide Transformer...")
    model_path = "./artifacts/nucleotide-transformer-500m-human-ref"
    nucleotide_transformer = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nucleotide_transformer.to(device)
    print(f"Model loaded on {device}.")
    

    # be sure to mask data using mask_data.py in preprocessing/GRCh38_hg38/
    data_path = "../preprocessing/GRCh38_hg38/artifacts/datasets/human_reference_genome/6kbp/masked"
    test_masked = {
        'random': os.path.join(data_path, "masked_test_random_ds"),
        'central': os.path.join(data_path, "masked_test_central_ds")
    }

    print("Loading Randomly Masked Data...")
    random_scheme = load_from_disk(test_masked['random'])
    print("Randomly Masked Data loaded.")
    print("Reproducing metrics for random masking scheme...")
    reproduce_nt_metrics(nucleotide_transformer, random_scheme, "nt-500m-hf_test_metrics_random.npz")

    print("Loading Central Masked Data...")
    central_scheme = load_from_disk(test_masked['central'])
    print("Central Masked Data loaded.")
    print("Reproducing metrics for central masking scheme...")
    reproduce_nt_metrics(nucleotide_transformer, central_scheme, "nt-500m-hf_test_metrics_central.npz")

    print("Done.")
