for d in 4 8 ; do 
	for glayers in 1 2 3; do 
		for residual in none linear dense ; do
			for p in blocks ferry gripper npuzzle ; do
				sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p $d $glayers $residual 
			done
		done
	done
done
