for glayers in 1 2; do 
	for d in 8 ; do 
		for residual in linear; do
			for loss in lstar l2 lrt lgbfs  ; do
				for arch in pddl asnet hgnn hgnnlite ; do
					for p in blocks ferry npuzzle gripper elevators_00 elevators_11; do
					# for p in spanner elevators_00 elevators_11; do
						sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p $d $glayers $residual $loss $arch
					done
				done
			done
		done
	done
done
