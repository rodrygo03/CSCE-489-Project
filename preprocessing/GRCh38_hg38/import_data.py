"""Import GRCh38/hg38 dataset"""

from datasets import load_dataset


# load from hugging face hub
#ds = load_dataset("InstaDeepAI/human_reference_genome", name="6kbp")
#print(ds)                 # shows train/validation/test sizes
#print(ds["train"][0])     # first window dict


# use InstaDeep's script
ds = load_dataset("human_reference_genome.py", name="6kbp")

#print(ds)
#print(ds["train"].features)     # sequence, chromosome, start_pos, end_pos
#print(ds["train"][0]["sequence"][:120])  # peek
print(ds.cache_files)

