using PDDL2Graph
using PDDL
using Flux
using GraphSignals
using GeometricFlux
using SymbolicPlanners
using Test


domain = load_domain("sokoban.pddl")
problem = load_problem("s1.pddl")

pddle = PDDLExtractor(domain, problem) 
state = initstate(domain, problem)

@testset "extraction and basic gradient" begin
	h₀ = PDDL2Graph.multigraph(pddle, state)
	m = MultiModel(h₀, 4, d -> Chain(Dense(d, 32,relu), Dense(32,32)))
	ps = Flux.params(m)
	gs = gradient(() -> sum(m(h₀)), ps)
	@test all(gs[p] !== nothing for p in ps)
end

#construct training set for L2 loss
@testset "testing concatenation for batching" begin 
	# get training example by running A* planner with h_add heuristic
	state = initstate(domain, problem)
	goal = PDDL.get_goal(problem)
	planner = AStarPlanner(HAdd())
	sol = planner(domain, state, goal)
	plan = collect(sol)
	trajectory = sol.trajectory
	satisfy(domain, sol.trajectory[end], goal)

	@testset "forward path" begin 
		h₀ = PDDL2Graph.multigraph(pddle, state)
		m = MultiModel(h₀, 4, d -> Chain(Dense(d, 32,relu), Dense(32,32)))
		xx = [PDDL2Graph.multigraph(pddle, state) for state in sol.trajectory];
		yy = collect(length(sol.trajectory):-1:1);
		@test reduce(hcat, map(m, xx)) ≈  m(reduce(cat, xx))
		ii = [7,1,6,2,5,3,4]
		@test reduce(hcat, map(m, xx[ii])) ≈  m(reduce(cat, xx[ii]))
	end

	@testset "gradient path" begin 
		xx = [PDDL2Graph.multigraph(pddle, state) for state in sol.trajectory];
		batch = reduce(cat, xx);
		yy = collect(length(sol.trajectory):-1:1);

		m = MultiModel(h₀, 4, d -> Chain(Dense(d, 32,relu), Dense(32,1)))
		ps = Flux.params(m);
		gs1 = gradient(ps) do 
			map(xx, yy) do h₀, y
				(sum(m(h₀)) - y)^2
			end |> sum 
		end;

		gs2 = gradient(ps) do 
			sum((vec(m(batch)) .- yy) .^ 2)
		end;
		@test all(gs1[p] ≈ gs2[p] for p in ps)
	end
end