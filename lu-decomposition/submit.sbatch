#!/bin/bash
#SBATCH --export=ALL
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:60:00 
#SBATCH --partition=scavenge

W=`grep processor /proc/cpuinfo | wc -l`
W=`expr ${W} + 0`
M=8000
lscpu
export KMP_AFFINITY=verbose # just to print the bindings
OMP_PROC_BIND=spread time ./lu-omp-serial $M 1
for ((i = ${W};  i >=1;  i/=2)); do 
  OMP_PROC_BIND=spread time ./lu-omp $M $i
done
