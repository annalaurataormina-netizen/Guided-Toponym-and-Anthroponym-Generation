#!/bin/bash
#SBATCH --output=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/toponyms/check.out

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/toponyms
python3 check.py
