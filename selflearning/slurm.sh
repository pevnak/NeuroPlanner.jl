#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH -p cpulong
#SBATCH --error=/home/pevnytom/logs/pddl.%j.err
#SBATCH --out=/home/pevnytom/logs/pddl.%j.out

ml Julia/1.8.3-linux-x86_64
julia -O3 --project=. supervised.jl $1 $6 $5 --dense-dim $2 --graph-layers $3  --residual $4 --seed $7
