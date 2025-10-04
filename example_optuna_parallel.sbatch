#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH
#SBATCH --partition main ### partition name where to run a job. Use 'main' unless qos is required. qos partitions 'rtx3090' 'rtx2080' 'gtx1080' 
#SBATCH --time 7-00:00:00
#SBATCH --job-name optuna_parallel_%a ### %a will be replaced with array task ID
#SBATCH --output optuna_parallel-%A_%a.out ### %A is job ID, %a is array task ID
#SBATCH --mail-user=razshmue@post.bgu.ac.il ### user's email for sending job status notifications
#SBATCH --mail-type=BEGIN,END,FAIL ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gpus=1 ### number of GPUs per task
#SBATCH --array=1-7 ### Run 7 parallel workers (adjust based on available GPUs)

### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID

### Start your code below ####
# Set environment variables
export SC2PATH="$HOME/StarCraftII"

# Activate virtual environment
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate safepo

which python
python --version
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__)"

# Go to project directory
cd /home/razshmue/workspace/Safe-Policy-Optimization-Modified

# Each array task will run the Optuna optimization
# Optuna will automatically coordinate between parallel workers
python safepo/multi_agent/run_optuns.py

# Deactivate the virtual environment
conda deactivate

echo "Job finished at: $(date)"