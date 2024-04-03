using NeuroPlanner
using Test 
using NeuroPlanner.Flux.Zygote

@testset "LazyVCatMatrix forward" begin
	for T in (Float32, Float64)
		dense_matrices = (randn(T,16,127), randn(T,4,127), randn(T, 8,127))
		dedu_matrices = (
			NeuroPlanner.DeduplicatedMatrix(randn(T,16,5), [rand(1:5) for _ in 1:127]),
			NeuroPlanner.DeduplicatedMatrix(randn(T,4,7), [rand(1:7) for _ in 1:127]),
			NeuroPlanner.DeduplicatedMatrix(randn(T,8,13), [rand(1:13) for _ in 1:127]),
			)

		for (x,y,z) in (dense_matrices, dedu_matrices)

			L = NeuroPlanner.LazyVCatMatrix((x,y,z))
			w = randn(T, 16,28)
			@test w * vcat(x,y,z) ≈ w * L;
			@test eltype(w * L) == T

			# test transposition
			ȳ = w * vcat(x,y,z)
			@test ȳ * vcat(x,y,z)' ≈ ȳ * L' rtol=1e-5

			L = NeuroPlanner.LazyVCatMatrix((x,y))
			w = randn(T, 16,20)
			@test eltype(w * L) == T
			@test w * vcat(x,y) ≈ w * L;

			# test transposition
			ȳ = w * vcat(x,y)
			@test ȳ * vcat(x,y)' ≈ ȳ * L'  rtol=1e-5

			L = NeuroPlanner.LazyVCatMatrix((x,))
			w = randn(T, 16,16)
			@test w * x ≈ w * L;
			@test eltype(w * L) == T

			# test transposition
			ȳ = w * x 
			@test ȳ * x' ≈ ȳ * L' rtol=1e-5
		end

	end
end

@testset "LazyVCatMatrix forward" begin
	for T in (Float32, Float64)
		dense_matrices = (randn(T,16,127), randn(T,4,127), randn(T, 8,127))
		dedu_matrices = (
			NeuroPlanner.DeduplicatedMatrix(randn(T,16,5), [rand(1:5) for _ in 1:127]),
			NeuroPlanner.DeduplicatedMatrix(randn(T,4,7), [rand(1:7) for _ in 1:127]),
			NeuroPlanner.DeduplicatedMatrix(randn(T,8,13), [rand(1:13) for _ in 1:127]),
			)

		for (x,y,z) in (dense_matrices, )
			L = NeuroPlanner.LazyVCatMatrix((x,y,z))
			w = randn(T, 16,28)
			ȳ = cos.(w*vcat(x,y,z))
			@test gradient(w -> sum(sin.(w*L)), w)[1] ≈ ȳ * vcat(x,y,z)' rtol = 1e-3
			δL = w' * ȳ
			@test gradient(L -> sum(sin.(w*L)), L)[1].xs[1] ≈ δL[1:16,:] rtol = 1e-3
			@test gradient(L -> sum(sin.(w*L)), L)[1].xs[2] ≈ δL[17:20,:] rtol = 1e-3
			@test gradient(L -> sum(sin.(w*L)), L)[1].xs[3] ≈ δL[21:28,:] rtol = 1e-3


			L = NeuroPlanner.LazyVCatMatrix((x,y))
			w = randn(T, 16,20)
			ȳ = cos.(w*vcat(x,y))
			@test gradient(w -> sum(sin.(w*L)), w)[1] ≈ ȳ * vcat(x,y)' rtol = 1e-3
			δL = w' * ȳ
			@test gradient(L -> sum(sin.(w*L)), L)[1].xs[1] ≈ δL[1:16,:] rtol = 1e-3
			@test gradient(L -> sum(sin.(w*L)), L)[1].xs[2] ≈ δL[17:20,:] rtol = 1e-3

			# test transposition
			L = NeuroPlanner.LazyVCatMatrix((x,y))
			w = randn(T, 16,20)
			ȳ = cos.(w*x)
			@test gradient(w -> sum(sin.(w*L)), w)[1] ≈ ȳ * x' rtol = 1e-3
			δL = w' * ȳ
			@test gradient(L -> sum(sin(w*L)), L)[1].xs[1] ≈ δL[1:16,:] rtol = 1e-3
		end

	end
end
