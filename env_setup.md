# **Environment Setup & Job Instructions**

## **1. Load Dependencies for the Job**

```bash
module purge
module load GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1
```

Save the loaded modules as a module set named **`deps`**:

```bash
module save deps
```

---

## **2. Create and Configure the Python Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate

pip install -U pip
pip install "datasets==3.6.0"
pip install --upgrade "transformers==4.38.0"
```

---

## **3. Development & Job Submission Notes**

When using **heavy Python imports** (e.g., `transformers`, `torch`), use a **SLURM job script** instead of running interactively.

Reference script:

```
/preprocessing/GRCh38_h38/preprocessing.slurm
```

### **Typical SLURM Script Structure**

```bash
#!/bin/bash
#SBATCH --time=HH:MM:SS
#SBATCH --cpus-per-task=1
#SBATCH --mem=XXG
#SBATCH --output=Output.%j

module restore deps   # restore saved module environment

# Optional: activate your virtual environment
# source venv/bin/activate
# pip install <extra-packages>

# Run your Python program
python file.py
```

---

## **4. Output Behavior**

SLURM generates an output file defined by:

```
--output=<filename>.%j
```

Where:

* **`%j`** = Job ID
* File contains **stdout** (and stderr unless redirected otherwise)

Example:

```
JobOutput.1234567
```

---

## **5. Installing Additional Packages**

To manually install more packages into your virtual environment:

```bash
source venv/bin/activate
pip install <package>
deactivate
```

Alternatively, you can install packages **inside your SLURM script** using:

```bash
source venv/bin/activate
pip install <package>
```

### **Reminder**

If your job relies on virtual environment packages, include:

```bash
source venv/bin/activate
```

inside the SLURM job script **before** executing Python.


