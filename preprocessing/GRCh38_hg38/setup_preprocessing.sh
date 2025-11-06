#!/bin/bash

# download data into 
mkdir -p "artifacts"

# give path to HF
HF_BASE_DIR="${PWD}/artifacts"

mkdir -p "${HF_BASE_DIR}/datasets"

export HF_HOME=${HF_BASE_DIR}
export HF_DATASETS_CACHE="${HF_BASE_DIR}/datasets"

echo "  HF_HOME=${HF_HOME}"
echo "  HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

