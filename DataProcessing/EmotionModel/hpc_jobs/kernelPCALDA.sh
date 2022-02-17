#!/bin/sh
#BSUB -q hpc
#BSUB -J KERN
### number of core
#BSUB -n 1
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=600GB]"
### Number of hours needed
#BSUB -W 23:59
### added outputs and errors to files
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Error_%J.err

echo "Runnin script..."

module load cuda/10.2
module load python3/3.8.11
python3 main_KernelPCA_LDA.py > performance/KERN_$(date +"%d-%m-%y")_$(date +'%H:%M:%S')
