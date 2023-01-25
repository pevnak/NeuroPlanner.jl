#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=cpulong
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --error=/home/pevnytom/logs/pddl.%j.err
#SBATCH --out=/home/pevnytom/logs/pddl.%j.out

ml Julia/1.8.3-linux-x86_64
export JULIA_PROJECT="/home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning"
# julia -O3 --project=. supervised.jl $1 $2 $3
julia -O3 --project=. investigate_regression.jl $1 $2 $3
