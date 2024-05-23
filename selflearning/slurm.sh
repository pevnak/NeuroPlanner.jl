#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1 --cores-per-socket=8
#SBATCH --mem=64G
#SBATCH -p amdextralong
#SBATCH --error=/home/pevnytom/logs/pddl.%j.err
#SBATCH --out=/home/pevnytom/logs/pddl.%j.out

ml Julia/1.9.3-linux-x86_64
julia --pkgimages=no --project=. supervised.jl $1 $6 $5 --dense-dim $2 --graph-layers $3 --aggregation meanmax  --residual $4 --seed $7
