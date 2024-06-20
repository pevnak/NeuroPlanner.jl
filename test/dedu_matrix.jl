using NeuroPlanner
using Test 
using NeuroPlanner.Flux 
using NeuroPlanner: DeduplicatedMatrix
using FiniteDifferences

@testset "find duplicates" begin
	max_val = 11
	for i in 1:10
		x = rand(1:max_val,100)
		m₁, ii₁ = NeuroPlanner.find_duplicates(x)
		m₂, ii₂ = NeuroPlanner.find_duplicates(x, max_val)
		@test m₁ == m₂
		@test ii₁ == ii₂
	end

	for i in 1:10
		x = [(rand(1:max_val), rand(1:max_val)) for _ in 1:100]
		m₁, ii₁ = NeuroPlanner.find_duplicates(hash.(x))
		m₂, ii₂ = NeuroPlanner.find_duplicates(x, max_val)
		@test m₁ == m₂
		@test ii₁ == ii₂
	end
	
	for i in 1:10
		x = (rand(1:max_val, 100), rand(1:max_val,100))
		m₁, ii₁ = NeuroPlanner.find_duplicates(hash.(zip(x...)))
		m₂, ii₂ = NeuroPlanner.find_duplicates(x, max_val)
		@test m₁ == m₂
		@test ii₁ == ii₂
	end
end

@testset "DeduplicatedMatrix" begin 
	x = [1 2 2 1; 3 4 4 5]
	dx = DeduplicatedMatrix(x)
	@test dx.x == [1 2 1; 3 4 5]
	@test dx.ii == [1, 2, 2, 3]
	@test Matrix(dx) == x
	@test Matrix(dx) isa Matrix


	dx = DeduplicatedMatrix(randn(Float32,3,5), [rand(1:5) for _ in 1:27])
	x = Matrix(dx)
	@test DeduplicatedMatrix(dx).x ≈ dx.x
	@test DeduplicatedMatrix(dx).ii ≈ dx.ii
	@test gradient(_x -> sum(sin.(DeduplicatedMatrix(_x, dx.ii))), dx.x)[1] ≈ grad(central_fdm(5,1), _x -> sum(sin.(DeduplicatedMatrix(_x, dx.ii))), dx.x)[1] rtol = 1e-3
	for m in (Dense(3,2), Chain(Dense(3,3),Dense(3,2)))
		@test m(dx) isa DeduplicatedMatrix
		@test m(dx) ≈ m(x)
		@test gradient(_x -> sum(sin.(m(DeduplicatedMatrix(_x, dx.ii)))), dx.x)[1] ≈ grad(central_fdm(5,1), _x -> sum(sin.(m(DeduplicatedMatrix(_x, dx.ii)))), dx.x)[1] rtol = 1e-3
		@test _isapprox(gradient(m -> sum(m(x)), m), gradient(m -> sum(m(dx)), m); rtol = 1e-3)
		@test _isapprox(gradient(m -> sum(sin.(m(x))), m), gradient(m -> sum(sin.(m(dx))), m); rtol = 1e-3)
		# @test _isapprox(Yota.grad(model -> sum(model(x)), m)[2][2],gradient(m -> sum(m(x)), m)[1])
	end
end

