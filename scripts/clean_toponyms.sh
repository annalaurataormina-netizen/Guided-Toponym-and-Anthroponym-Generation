#!/bin/bash
#SBATCH --output=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/toponyms/clean.out

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/toponyms
python3 clean.py
