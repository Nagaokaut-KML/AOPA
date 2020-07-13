#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=15:30:00
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh
module load singularity/2.6.1
cd /home/acb11319bq/ito/proj_aopa/proj_aopa/ito
singularity exec -B /groups2/gcb50278 --nv ./medical.img python3 train_class.py