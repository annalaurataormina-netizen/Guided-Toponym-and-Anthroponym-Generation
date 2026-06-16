#!/bin/bash
#SBATCH --output=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/extract_%j.out
#SBATCH --error=/vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation/extract_%j.err

cd /vol/bitbucket/at2225/Guided-Toponym-and-Anthroponym-Generation
python3 extract.py
