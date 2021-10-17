#!/bin/bash
#SBATCH --export=ALL
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:60:00 
#SBATCH --partition=scavenge

export KMP_AFFINITY=verbose # just to print the bindings
:: OMP_PROC_BIND=spread time ./lu-omp-serial 7000 1
for ((i = 16; i <= 32; i+=16)); do 
  OMP_PROC_BIND=spread time ./lu-omp 7000 $i
done
