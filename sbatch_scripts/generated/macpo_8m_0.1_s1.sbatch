#!/bin/bash
#SBATCH --partition=rtx3090
#SBATCH --qos=razshmue
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

cd /home/corsound/workspace/Safe-Policy-Optimization

PYTHONPATH=. python safepo/multi_agent/macpo.py \
    --env-name 8m \
    --cost-type collision \
    --cost-limit 0.1 \
    --seed 1 \
    --write_terminal \
    --total-steps 10000000

conda deactivate
echo "Job finished at: $(date)"
