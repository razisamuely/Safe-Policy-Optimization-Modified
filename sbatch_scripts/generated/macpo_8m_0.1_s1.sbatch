#!/bin/bash
#SBATCH --partition=rtx3090
#SBATCH --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --job-name=macpo_8m_0.1_s1
#SBATCH --output=sbatch_scripts/generated/macpo_8m_0.1_s1-id-%J.out
#SBATCH --mail-user=razshmue@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=1

export SC2PATH="$HOME/StarCraftII"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate safepo

### Start GPU Keep-Alive in background ###
cd /home/razshmue/workspace/Safe-Policy-Optimization-Modified
python sbatch_scripts/gpu_keepalive.py > sbatch_scripts/generated/gpu_keepalive_${SLURM_JOBID}.log 2>&1 &
KEEPALIVE_PID=$!

# Function to cleanup keep-alive on exit
cleanup() {
    if kill -0 $KEEPALIVE_PID 2>/dev/null; then
        kill $KEEPALIVE_PID
    fi
}
trap cleanup EXIT

PYTHONPATH=. python safepo/multi_agent/macpo.py \
    --task 8m \
    --cost-type collision \
    --cost-limit 0.1 \
    --seed 1 \
    --write-terminal True \
    --num-envs 5 \
    --total-steps 10000000

conda deactivate
echo "Job finished at: $(date)"
