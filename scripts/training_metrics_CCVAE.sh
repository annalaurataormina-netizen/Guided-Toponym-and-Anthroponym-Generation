#!/bin/bash
#SBATCH --out=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/CCVAE/out/training_metrics_%j.out
#SBATCH --error=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/CCVAE/err/training_metrics_%j.err

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/
python3 -m CCVAE.training_metrics
