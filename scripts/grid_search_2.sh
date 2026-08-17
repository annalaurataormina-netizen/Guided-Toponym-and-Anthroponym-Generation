#!/bin/bash
#SBATCH --job-name=lr_grid
#SBATCH --out=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/CCVAE/out/grid_search_%A_%a.out
#SBATCH --error=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/CCVAE/err/grid_search_%A_%a.err
#SBATCH --array=4-9%

LEARNING_RATES=(
    0.0001
    0.0005
    0.00075
    0.001
    0.0015
    0.002
    0.0025
    0.003
    0.005
    0.01
)

LEARNING_RATE=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/

echo "Running learning rate = $LEARNING_RATE"

source venv/bin/activate

python3 -m CCVAE.grid_search --lr "$LEARNING_RATE"
