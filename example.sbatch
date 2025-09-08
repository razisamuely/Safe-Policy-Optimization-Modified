#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH
#SBATCH --partition main ### partition name where to run a job. Use 'main' unless qos is required. qos partitions 'rtx3090' 'rtx2080' 'gtx1080' 
#SBATCH --time 0-10:30:00 ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name smac_test_cost_term_3s_vs_4z_combined ### name of the job. replace my_job with your desired job name
#SBATCH --output smac_test_cost_term_3s_vs_4z_combined-id-%J.out ### output log for running job - %J is the job number variable
#SBATCH --mail-user=razshmue@post.bgu.ac.il ### user's email for sending job status notifications
#SBATCH --mail-type=BEGIN,END,FAIL ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gpus=1 ### number of GPUs. Choosing type e.g.: #SBATCH --gpus=gtx_1080:1 , or rtx_2080, or rtx_3090 . Allocating more than 1 requires the IT team's permission
##SBATCH --tasks=1 # 1 process – use for processing of few programs concurrently in a job (with srun). Use just 1 otherwise

### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

### Start your code below ####
module load anaconda ### load anaconda module

# Set environment variables
export SC2PATH="$HOME/StarCraftII"

# Activate virtual environment
conda activate safepo

# Go to project directory
cd /home/razshmue/workspace/Safe-Policy-Optimization-Modified

# Run the training script
python safepo/multi_agent/macpo.py --task 3m --total-steps 10000 --cost-type dead_allies

# Deactivate the virtual environment
conda deactivate

echo "Job finished at: $(date)"