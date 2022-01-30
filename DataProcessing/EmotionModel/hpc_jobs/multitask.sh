#!/bin/sh
#BSUB -q gpua100
#BSUB -J MULT
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

module load cuda/10.2
module load python3/3.8.11
python3 main_MULT.py > performance/multitask_$(date +"%d-%m-%y")_$(date +'%H:%M:%S')
