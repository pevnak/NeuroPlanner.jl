# submitting supervised
# ARCHS = "pddl asnet hgnn hgnnlite lrnn  mixedlrnn"
# LOSSES = "lstar l2 lrt lgbfs"
LOSSES="lstar l2"
ARCHS="lrnn mixedlrnn"
PROBLEMS="spanner blocks ferry npuzzle gripper elevators_00"

for seed in 2 3 ; do 
	for glayers in 1 2 3; do 
		for d in 4 8 16 ; do 
			for residual in none linear; do
				for loss in lstar l2 ; do
					for arch in $ARCHS ; do
						for p in $PROBLEMS ; do						
							sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p $d $glayers $residual $loss $arch $seed
						done
					done
				done
			done
		done
	done
done