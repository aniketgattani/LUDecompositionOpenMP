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
echo $W
lscpu
::export KMP_AFFINITY=verbose # just to print the bindings
OMP_PLACES="{0:1}:${W}:1" OMP_PROC_BIND=spread time ./lu-omp-serial 7000 1
for ((i = 2; i <= 32; i*=2)); do 
  OMP_PLACES="{0:1}:${W}:1" OMP_PROC_BIND=spread time ./lu-omp 7000 $i
done
OMP_PLACES="{0:1}:${W}:1" OMP_PROC_BIND=spread time ./lu-omp 7000 48
