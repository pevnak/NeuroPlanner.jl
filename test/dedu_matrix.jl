using NeuroPlanner
using Test 
using Flux 
using NeuroPlanner: DeduplicatedMatrix

@testset "DeduplicatedMatrix" begin 
	x = [1 2 2 1; 3 4 4 5]
	dx = DeduplicatedMatrix(x)
	@test dx.x == [1 2 1; 3 4 5]
	@test dx.ii == [1, 2, 2, 3]
	@test Matrix(dx) == x
	@test Matrix(dx) isa Matrix

	x = Float32[1 2 2 1; 3 4 4 5]
	dx = DeduplicatedMatrix(x)
	for m in (Dense(2,3), Chain(Dense(2,3),Dense(3,2)))
		@test m(dx) isa DeduplicatedMatrix
		@test m(dx) ≈ m(x)
		@test _isapprox(gradient(m -> sum(m(x)), m)[1], gradient(m -> sum(m(dx)), m)[1])
	end
end

