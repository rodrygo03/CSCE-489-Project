Environment Setup & Job Instructions
1. Load Dependencies for Job
module purge
module load GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1
Save the loaded modules as a module set named deps:
module save deps
2. Create and Configure Python Virtual Environment
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install datasets
pip install --upgrade "transformers==4.38.0"
3. Notes for Development & Job Submission
When using heavy Python imports (e.g., transformers, torch), create a .slurm job script.
For reference, see:
/preprocessing/GRCh38_h38/preprocessing.slurm
A typical .slurm script follows this structure:
# Resource allocation directives (--time, --cpus-per-task, --mem, etc.)

module restore deps    # restores saved module environment

# Optional: activate venv and install additional dependencies
# source venv/bin/activate
# pip install <extra-packages>

# Execute your Python script
python file.py
4. Output Behavior
SLURM creates an output file named:
<--output>.%j
Where %j is the JobID.
This file contains the stdout of the job.
5. Installing Additional Packages
To install more packages manually:
source venv/bin/activate
pip install <package>
deactivate
Alternatively, include pip install commands directly inside your .slurm script.
Reminder:
If your job uses virtual-environment packages, include:
source venv/bin/activate
inside your .slurm script.
