
# load dependecies to use in job
module purge 
module load GCC/12.3.0  OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1

# save list as deps (we will use this var later...)
module save deps

# create venv to get additonal deps
python -m venv venv
source venv/bin/activate
pip install -U pip

pip install datasets
pip install --upgrade "transformers==4.38.0"

# later when dev and using heavy imports like transformers/torch
# create a .slurm file similar to bash follow /preprocessing/GRCh38_h38/preprocessing.slurm for reference
# it follows the following general structure
#
# resource allocation
# load modules via "module restore deps"
# other logic if needed.
# python file.py
#
# "output" will be a <--output>.<JobID> file that contains stdout of the job

# To install more packages manually source venv/bin/activate and then pip install what is needed. then 
# deactivate and run job. Or include pip install what is needed in the job script itself
