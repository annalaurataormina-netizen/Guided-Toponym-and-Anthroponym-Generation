#!/bin/bash
#SBATCH --out=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/toponyms/out/check_%j.out
#SBATCH --error=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/toponyms/err/check_%j.err

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/toponyms
python3 check.py
