#!/bin/bash
#SBATCH --out=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/out/temp_%j.out
#SBATCH --error=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/err/temp_%j.err

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation
python3 temp.py
