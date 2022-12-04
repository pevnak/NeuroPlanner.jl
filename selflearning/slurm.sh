#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --error=/home/tomas.pevny/logs/pddl.%j.err
#SBATCH --out=/home/tomas.pevny/logs/pddl.%j.out
/home/tomas.pevny/julia-1.8.0/bin/julia -O3 --project=. self_learning.jl $1 $2 $3
