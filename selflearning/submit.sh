for ((i = 11;i<=100;i++)) ; do
	for p in blocks ferry gripper npuzzle ; do
		# sbatch -p high -D /home/tomas.pevny/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p $i
		sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p $i
	done
done
