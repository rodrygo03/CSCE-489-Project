#!/bin/bash

# Script to download nucleotide-transformer model from hugging face

set -e  

MODEL_NAME="nucleotide-transformer-500m-human-ref"
BASE_URL="https://huggingface.co/InstaDeepAI/${MODEL_NAME}/resolve/main"
MODEL_DIR="artifacts/${MODEL_NAME}"

echo "Creating model directory..."
mkdir -p "${MODEL_DIR}"

echo "Downloading model files to ${MODEL_DIR}..."
cd "${MODEL_DIR}"

echo "Downloading config files..."
wget -c "${BASE_URL}/config.json"
wget -c "${BASE_URL}/tokenizer_config.json"
wget -c "${BASE_URL}/vocab.txt"
wget -c "${BASE_URL}/special_tokens_map.json"

echo "Downloading model weights (this may take a while)..."
wget -c "${BASE_URL}/pytorch_model.bin"

# Optional: Download additional files if they exist
echo "Downloading additional files..."
wget -c "${BASE_URL}/tokenizer.json" 2>/dev/null || echo "tokenizer.json not found, skipping..."

cd -

echo "**Done**" 
echo "Model saved to ${MODEL_DIR}"
echo ""
echo "Load model:"
echo "  tokenizer = AutoTokenizer.from_pretrained('${MODEL_DIR}', local_files_only=True)"
echo "  model = AutoModelForMaskedLM.from_pretrained('${MODEL_DIR}', local_files_only=True)"