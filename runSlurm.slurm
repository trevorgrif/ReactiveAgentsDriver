#!/bin/bash
#SBATCH -J ABM
#SBATCH --mem=180G
#SBATCH --cpus-per-task=51
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EMAIL
#SBATCH -C "ceph"

. /usr/modules/init/bash
module load julia

cd $SLURM_SUBMIT_DIR

julia -t 51 $FILE
