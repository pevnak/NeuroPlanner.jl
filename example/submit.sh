# for p in zenotravel ; do
for p in blocks ferry gripper npuzzle ; do
	for l in l2 lstar; do 
		sbatch -D /home/tomas.pevny/julia/Pkg/PDDL2Graph.jl/example slurm.sh $p $l 3
	done
done