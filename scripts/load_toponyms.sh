#!/bin/bash
#SBATCH --out=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/db/out/load_toponyms_%j.out
#SBATCH --error=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/db/err/load_toponyms_%j.err

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/db
python3 toponyms.py