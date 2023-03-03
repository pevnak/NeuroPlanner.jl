using NeuroPlanner
using PDDL
using Flux
using GraphNeuralNetworks
using SymbolicPlanners
using Test
using Random

# domain = load_domain("sokoban.pddl")
# problem = load_problem("s1.pddl")

domain = load_domain("../classical-domains/classical/settlers/domain.pddl")
problem = load_problem("../classical-domains/classical/settlers/p01_pfile1.pddl")


domain = load_domain("../classical-domains/classical/depot/domain.pddl")
problem = load_problem("../classical-domains/classical/depot/pfile1.pddl")

domain = load_domain("../classical-domains/classical/driverlog/domain.pddl")
problem = load_problem("../classical-domains/classical/driverlog/pfile1.pddl")

domain = load_domain("../classical-domains/classical/briefcaseworld/domain.pddl")
problem = load_problem("../classical-domains/classical/briefcaseworld/pfile1.pddl")


# pddle = PDDLExtractor(domain, problem) 

pddle = PDDLExtractor(domain, problem) 
state = initstate(domain, problem)

@testset "extraction and basic gradient" begin
	h₀ = pddle(state)
	m = MultiModel(h₀, 4, 2, d -> Chain(Dense(d, 32,relu), Dense(32,32)))
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
		h₀ = pddle(state)
		m = MultiModel(h₀, 4, 2, d -> Chain(Dense(d, 32,relu), Dense(32,32)))
		xx = [pddle(state) for state in sol.trajectory];
		yy = collect(length(sol.trajectory):-1:1);
		@test reduce(hcat, map(m, xx)) ≈  m(batch(xx))
		ii = randperm(length(xx))
		@test reduce(hcat, map(m, xx[ii])) ≈  m(batch(xx[ii]))
	end

	@testset "gradient path" begin 
		h₀ = pddle(state)
		m = MultiModel(h₀, 4, 2, d -> Chain(Dense(d, 32,relu), Dense(32,1)))
		xx = [pddle(state) for state in sol.trajectory];
		bxx = batch(xx);
		yy = collect(length(sol.trajectory):-1:1);

		ps = Flux.params(m);
		gs1 = gradient(ps) do 
			map(xx, yy) do h₀, y
				(sum(m(h₀)) - y)^2
			end |> sum 
		end;

		gs2 = gradient(ps) do 
			sum((vec(m(bxx)) .- yy) .^ 2)
		end;
		@test all(maximum(abs2.(gs1[p] .- gs2[p])) < 1e-6 for p in ps)
	end
end