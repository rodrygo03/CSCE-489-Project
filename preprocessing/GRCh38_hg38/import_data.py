"""Import GRCh38/hg38 dataset"""

from datasets import load_dataset

# use InstaDeep's script to obatain data
ds = load_dataset("human_reference_genome.py", name="6kbp")

print("**DONE**")
print(ds.cache_files)

