for glayers in 1 2 3; do 
	for d in 4 8 16 ; do 
		for residual in none linear; do
			for loss in lstar l2 ; do
				for arch in pddl asnet ; do
					for p in blocks ferry gripper npuzzle ; do
						sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p $d $glayers $residual $loss $arch
					done
				done
			done
		done
	done
done
