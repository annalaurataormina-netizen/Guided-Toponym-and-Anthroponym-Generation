#!/bin/bash
#SBATCH --out=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/VAE/out/train_%j.out
#SBATCH --error=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/VAE/err/train_%j.err

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/
python3 -m VAE.train
