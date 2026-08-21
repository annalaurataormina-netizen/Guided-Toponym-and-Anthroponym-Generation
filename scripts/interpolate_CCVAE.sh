#!/bin/bash
#SBATCH --out=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/CCVAE/out/interpolate_%j.out
#SBATCH --error=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/CCVAE/err/interpolate_%j.err

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/
python3 -m CCVAE.interpolate
