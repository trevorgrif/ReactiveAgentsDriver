#!/bin/bash
#SBATCH -J ABM
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EMAIL
#SBATCH -C "ceph"

. /usr/modules/init/bash
module load julia

cd $SLURM_SUBMIT_DIR

julia -t 10 $FILE
