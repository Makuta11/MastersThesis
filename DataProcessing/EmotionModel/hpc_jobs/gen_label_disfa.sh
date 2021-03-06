#!/bin/sh
#BSUB -q hpc
#BSUB -J LabelGeneration
### number of core
#BSUB -n 24
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=5GB]"
### Number of hours needed
#BSUB -W 23:59
### added outputs and errors to files
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Error_%J.err

echo "Runnin script..."

module load python3/3.8.11
python3 src/disfa_label_generation.py > outputs/log_file_disfa_datagen

