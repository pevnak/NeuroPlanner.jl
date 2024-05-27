using Test
using NeuroPlanner

@testset "_inlined_search" begin 
	@test NeuroPlanner._inlined_search(:a, (:b,:a,:c)) == 2
	@test NeuroPlanner._inlined_search(:e, (:b,:a,:c)) == -1
end
