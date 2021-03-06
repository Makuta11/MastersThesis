#!/bin/sh
#BSUB -q gpuv100
#BSUB -J SINGLE
### number of core
#BSUB -n 1
### specify that all cores should be on the same host
#BSUB -gpu "num=1:mode=exclusive_process"
### specify the memory needed
#BSUB -R "rusage[mem=50GB]"
### Number of hours needed
#BSUB -W 23:59
### added outputs and errors to files
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Error_%J.err

echo "Runnin script..."

module load cuda/11.3
module load python3/3.8.11
python3 main_SINGLELAB.py > performance/SINGLAB$(date +"%d-%m-%y")_$(date +'%H:%M:%S')
