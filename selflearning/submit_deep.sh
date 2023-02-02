for ((i = 1;i<=3;i++)) ; do
	for p in blocks ferry gripper npuzzle ; do
		# sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning deep_slurm.sh $p lstar $i
		# sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning deep_slurm.sh $p lgbfs $i
		# sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning deep_slurm.sh $p l2 $i
		# sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning deep_slurm.sh $p lrt $i

		# sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning deep_slurm.sh $p both $i
		# sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning deep_slurm.sh $p forward $i
		# sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning deep_slurm.sh $p backward $i

		sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning deep_slurm.sh $p lstar $i
		# sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning deep_slurm.sh $p forward $i
		# sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning deep_slurm.sh $p backward $i


	done
done
