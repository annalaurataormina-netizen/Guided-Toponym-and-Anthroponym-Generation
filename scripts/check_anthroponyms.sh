#!/bin/bash
#SBATCH --output=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/anthroponyms/check.out

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/anthroponyms
python3 check.py
