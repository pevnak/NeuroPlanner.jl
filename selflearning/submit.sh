for ((i = 1;i<=1;i++)) ; do
	# for p in blocks  ; do
	# for p in blocks ferry gripper npuzzle ; do
	for p in blocks; do
		# for h in HAdd HMax null ; do
		for h in null ; do
			sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p lstar $h $i
			# sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p lgbfs $h $i
			sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p l2 $h $i
			sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p lrt $h $i
		done
	done
done
