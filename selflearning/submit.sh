# submitting supervised
# for seed in 2 3 ; do 
# 	for glayers in 1 2 3; do 
# 		for d in 4 8 16 ; do 
# 			for residual in none linear; do
# 				for loss in lstar l2 lrt lgbfs  ; do
# 					# for arch in pddl asnet hgnn hgnnlite ; do
# 					for arch in hgnn ; do
# 						# for p in spanner blocks ferry npuzzle gripper elevators_00; do						
# 						for p in gripper; do
# 							sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p $d $glayers $residual $loss $arch $seed
# 						done
# 					done
# 				done
# 			done
# 		done
# 	done
# done

# submitting supervised
for seed in 1 2 3 ; do 
	for glayers in 1 2 3; do 
		for d in 4 8 16 ; do 
			for residual in none; do
				for loss in bellman  ; do
					# for arch in pddl asnet hgnn hgnnlite ; do
					for arch in hgnn ; do
						for p in spanner blocks ferry npuzzle gripper elevators_00; do						
						# for p in gripper; do
							sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p $d $glayers $residual $loss $arch $seed
						done
					done
				done
			done
		done
	done
done


# submitting supervised
for seed in 1 2 3 ; do 
	for glayers in 1 2 3; do 
		for d in 4 8 16 ; do 
			for residual in none; do
				for loss in levinloss  ; do
					# for arch in pddl asnet hgnn hgnnlite ; do
					for arch in levinasnet ; do
						for p in spanner blocks ferry npuzzle gripper elevators_00; do						
						# for p in gripper; do
							sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $p $d $glayers $residual $loss $arch $seed
						done
					done
				done
			done
		done
	done
done


# submitting self-learning
# for seed in 1 2 3 ; do 
# 	for loss in lstar l2  ; do
# 		for p in spanner blocks ferry npuzzle gripper elevators_00 elevators_11; do						
# 			sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm_self.sh $p hgnn $loss $seed astar
# 		done
# 	done
# done


# submitting self-learning
for seed in 1 2 3 ; do 
	for p in spanner blocks ferry npuzzle gripper elevators_00; do						
		sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm_self.sh $p levinasnet levinloss $seed bfs
	done
done