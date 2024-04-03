using NeuroPlanner
using Test 

@testset "LazyVCatMatrix" begin
	for T in (Float32, Float64)
		dense_matrices = (randn(T,16,127), randn(T,4,127), randn(T, 8,127))
		dedu_matrices = (
			NeuroPlanner.DeduplicatedMatrix(randn(T,16,5), [rand(1:5) for _ in 1:127]),
			NeuroPlanner.DeduplicatedMatrix(randn(T,4,7), [rand(1:7) for _ in 1:127]),
			NeuroPlanner.DeduplicatedMatrix(randn(T,8,13), [rand(1:13) for _ in 1:127]),
			)

		for (x,y,z) in (dense_matrices, dedu_matrices)
			w = randn(T, 16,28)
			@test w * vcat(x,y,z) ≈ w * NeuroPlanner.LazyVCatMatrix((x,y,z));
			@test eltype(w * NeuroPlanner.LazyVCatMatrix((x,y,z))) == T
			w = randn(T, 16,20)
			@test w * vcat(x,y) ≈ w * NeuroPlanner.LazyVCatMatrix((x,y));
			w = randn(T, 16,16)
			@test w * x ≈ w * NeuroPlanner.LazyVCatMatrix((x,));
			@test eltype(w * NeuroPlanner.LazyVCatMatrix((x,))) == T
		end
	end
end
