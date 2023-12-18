#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=felix.burr@uni-ulm.de
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --export=CONFIG,ALL,EXECUTABLE="time python3.9 -u run_config.py"
#SBATCH -J OpenMP_Test
#Usually you should set
export KMP_AFFINITY=compact,1,0
#export KMP_AFFINITY=verbose,compact,1,0 prints messages concerning the supported affinity
#KMP_AFFINITY Description: https://software.intel.com/en-us/node/524790#KMP_AFFINITY_ENVIRONMENT_VARIABLE

export OMP_NUM_THREADS=$((${SLURM_JOB_CPUS_PER_NODE}/2))
startexe="${EXECUTABLE} ${CONFIG}"
python3.9 -c "import os; print(\"Threads: \" + str(len(os.sched_getaffinity(0))))"
echo $startexe
exec $startexe
