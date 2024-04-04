using NeuroPlanner
using Test 
using NeuroPlanner.Flux.Zygote
using FiniteDifferences
using NeuroPlanner: LazyVCatMatrix, DeduplicatedMatrix

@testset "LazyVCatMatrix forward" begin
	for T in (Float32, Float64)
		dense_matrices = (randn(T,16,127), randn(T,4,127), randn(T, 8,127))
		dedu_matrices = (
			DeduplicatedMatrix(randn(T,16,5), [rand(1:5) for _ in 1:127]),
			DeduplicatedMatrix(randn(T,4,7), [rand(1:7) for _ in 1:127]),
			DeduplicatedMatrix(randn(T,8,13), [rand(1:13) for _ in 1:127]),
			)

		for (x,y,z) in (dense_matrices, dedu_matrices)
			L = LazyVCatMatrix((x,y,z))
			w = randn(T, 16,28)
			@test w * vcat(x,y,z) ≈ w * L;
			@test eltype(w * L) == T

			# test transposition
			ȳ = w * vcat(x,y,z)
			@test ȳ * vcat(x,y,z)' ≈ ȳ * L' rtol=1e-5

			L = LazyVCatMatrix((x,y))
			w = randn(T, 16,20)
			@test eltype(w * L) == T
			@test w * vcat(x,y) ≈ w * L;

			# test transposition
			ȳ = w * vcat(x,y)
			@test ȳ * vcat(x,y)' ≈ ȳ * L'  rtol=1e-5

			L = LazyVCatMatrix((x,))
			w = randn(T, 16,16)
			@test w * x ≈ w * L;
			@test eltype(w * L) == T

			# test transposition
			ȳ = w * x 
			@test ȳ * x' ≈ ȳ * L' rtol=1e-5
		end

	end
end

@testset "LazyVCatMatrix backward" begin
	@testset "Dense" begin
		for T in (Float32, Float64)
			dense_matrices = (randn(T,16,127), randn(T,4,127), randn(T, 8,127))
			(x,y,z) = dense_matrices
			L = LazyVCatMatrix((x,y,z))
			w = randn(T, 16,28)
			@test gradient(w -> sum(sin.(w*L)), w)[1] ≈ grad(central_fdm(5, 1), w -> sum(sin.(w*L)), w)[1] rtol = 1e-3
			δL = gradient(L -> sum(sin.(w*L)), L)[1]
			@test δL.xs[1] ≈ grad(central_fdm(5, 1), x -> sum(sin.(w*LazyVCatMatrix((x,y,z)))), x)[1] rtol = 1e-3 
			@test δL.xs[2] ≈ grad(central_fdm(5, 1), y -> sum(sin.(w*LazyVCatMatrix((x,y,z)))), y)[1] rtol = 1e-3 
			@test δL.xs[3] ≈ grad(central_fdm(5, 1), z -> sum(sin.(w*LazyVCatMatrix((x,y,z)))), z)[1] rtol = 1e-3 

			L = LazyVCatMatrix((x,y))
			w = randn(T, 16,20)
			@test gradient(w -> sum(sin.(w*L)), w)[1] ≈ grad(central_fdm(5, 1), w -> sum(sin.(w*L)), w)[1] rtol = 1e-3
			δL = gradient(L -> sum(sin.(w*L)), L)[1]
			@test δL.xs[1] ≈ grad(central_fdm(5, 1), x -> sum(sin.(w*LazyVCatMatrix((x,y)))), x)[1] rtol = 1e-3 
			@test δL.xs[2] ≈ grad(central_fdm(5, 1), y -> sum(sin.(w*LazyVCatMatrix((x,y)))), y)[1] rtol = 1e-3 

			L = NeuroPlanner.LazyVCatMatrix((x,))
			w = randn(T, 16,16)
			@test gradient(w -> sum(sin.(w*L)), w)[1] ≈ grad(central_fdm(5, 1), w -> sum(sin.(w*L)), w)[1] rtol = 1e-3
			δL = gradient(L -> sum(sin.(w*L)), L)[1]
			@test δL.xs[1] ≈ grad(central_fdm(5, 1), x -> sum(sin.(w*LazyVCatMatrix((x,)))), x)[1] rtol = 1e-3 
		end
	end

	@testset "DeduplicatedMatrix" begin
		for T in (Float32, Float64)
			x = DeduplicatedMatrix(randn(T,16,5), [rand(1:5) for _ in 1:127])
			y = DeduplicatedMatrix(randn(T,4,7), [rand(1:7) for _ in 1:127])
			z = DeduplicatedMatrix(randn(T,8,13), [rand(1:13) for _ in 1:127])

			L = LazyVCatMatrix((x,y,z))
			w = randn(T, 16,28)
			@test gradient(w -> sum(sin.(w*L)), w)[1] ≈ grad(central_fdm(5, 1), w -> sum(sin.(w*L)), w)[1] rtol = 1e-3
			δL = gradient(L -> sum(sin.(w*L)), L)[1]
			@test δL.xs[1].x ≈ grad(central_fdm(5, 1), _x -> sum(sin.(w*LazyVCatMatrix((DeduplicatedMatrix(_x, x.ii),y,z)))), x.x)[1] rtol = 1e-3 
			@test δL.xs[2].x ≈ grad(central_fdm(5, 1), _y -> sum(sin.(w*LazyVCatMatrix((x,DeduplicatedMatrix(_y, y.ii),z)))), y.x)[1] rtol = 1e-3 
			@test δL.xs[3].x ≈ grad(central_fdm(5, 1), _z -> sum(sin.(w*LazyVCatMatrix((x,y,DeduplicatedMatrix(_z, z.ii))))), z.x)[1] rtol = 1e-3 

			L = LazyVCatMatrix((x,y))
			w = randn(T, 16,20)
			@test gradient(w -> sum(sin.(w*L)), w)[1] ≈ grad(central_fdm(5, 1), w -> sum(sin.(w*L)), w)[1] rtol = 1e-3
			δL = gradient(L -> sum(sin.(w*L)), L)[1]
			@test δL.xs[1].x ≈ grad(central_fdm(5, 1), _x -> sum(sin.(w*LazyVCatMatrix((DeduplicatedMatrix(_x, x.ii),y)))), x.x)[1] rtol = 1e-3 
			@test δL.xs[2].x ≈ grad(central_fdm(5, 1), _y -> sum(sin.(w*LazyVCatMatrix((x,DeduplicatedMatrix(_y, y.ii))))), y.x)[1] rtol = 1e-3 

			L = NeuroPlanner.LazyVCatMatrix((x,))
			w = randn(T, 16,16)
			@test gradient(w -> sum(sin.(w*L)), w)[1] ≈ grad(central_fdm(5, 1), w -> sum(sin.(w*L)), w)[1] rtol = 1e-3
			δL = gradient(L -> sum(sin.(w*L)), L)[1]
			@test δL.xs[1].x ≈ grad(central_fdm(5, 1), _x -> sum(sin.(w*LazyVCatMatrix((DeduplicatedMatrix(_x, x.ii),)))), x.x)[1] rtol = 1e-3 
		end
	end
end
