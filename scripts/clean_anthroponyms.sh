#!/bin/bash
#SBATCH --output=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/anthroponyms/clean_%j.out
#SBATCH --error=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/anthroponyms/clean_%j.err

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/anthroponyms
python3 clean.py
