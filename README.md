# CSCE-489 Semester Project
## Hybrid N-gram and Neural Language Model for Genomic Sequences

### General Usage
To run any specific `*.py` file, refer to its corresponding `*.slurm` file. Feel free to use the /logs dirs scattered throughout the repo for
the stdout of slurm jobs.

## HPC Setup
```bash
module purge
module load GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1
module save deps
```

## Venv Setup
```bash
python -m venv venv
source venv/bin/activate

pip install -U pip
pip install datasets
pip install --upgrade "transformers==4.38.0"
```
may be missing some...
if so 
```bash
module restore deps
source venv/bin/activate
pip install package
```
IMPORTANT!!!: *.slurm jobs that use the downloaded data have a path 
relative to preprocessing/GRCh38_hg38/artifacts/datasets/human_reference_genome/6kbp/0.0.0/relativeID/

relativeID needs to be changed according to your path

---

## `/infini-gram` (cloned repository)

### Modifications
- Only one file was changed from the source code:
  - **`/pkg/infini_gram/indexing.py`** — modified to support the nucleotide transformer tokenizer.

### Documentation & Build Notes
1. **`/pkg/indexing_notes.md`** — notes on building the Rust binary for indexing.
2. **Slurm scripts** — indicate how the Infini-Gram indices were built.
3. **`/pkg/compile_engine.slurm`** — compiles the query engine (last step before using Infini-Gram elsewhere).
4. **`test_indices.*`** — quick test for verifying index functionality.

---

## `/main`

### `nucleotide-transformer/`
Benchmarking baseline model performance.

### `intrinsic/`
- `eval_nt_metrics.py` — intrinsic evaluation script.

### `extrinsic/`
- `fine_tune/` — contains DeePromoter data, preprocessing scripts, and the fine-tuning pipeline.

---

## `/preprocessing/`

Scripts for obtaining data, masking, and converting formats.

### Important
Before running `import_data.py`, execute:

```bash
source setup_preprocessing.sh
```
This ensures data is downloaded into pwd (within scratch and not home)
