#!/bin/bash
#SBATCH --out=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/AE/out/test_%j.out
#SBATCH --error=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/AE/err/test_%j.err

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/AE
python3 test.py
