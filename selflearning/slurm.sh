#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -p amd
#SBATCH --error=/home/pevnytom/logs/pddl.%j.err
#SBATCH --out=/home/pevnytom/logs/pddl.%j.out

### /home/tomas.pevny/julia-1.8.0/bin/julia -O3 --project=. self_learning.jl $1 $2


ml Julia/1.8.3-linux-x86_64
julia -O3 --project=. potential_heuristic.jl $1 $2 $3
