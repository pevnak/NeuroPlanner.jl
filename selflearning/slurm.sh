#!/bin/bash
#SBATCH --time=42-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=16 --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -p amdextralong
#SBATCH --error=/home/pevnytom/logs/pddl.%j.err
#SBATCH --out=/home/pevnytom/logs/pddl.%j.out

ml Julia/1.9.3-linux-x86_64
julia --pkgimages=no --project=. supervised.jl $1 $6 $5 --dense-dim $2 --graph-layers $3  --residual $4 --seed $7
