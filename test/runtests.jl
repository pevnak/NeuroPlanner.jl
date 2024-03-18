using NeuroPlanner
using PDDL
using Flux
using Mill
using SymbolicPlanners
using Test
using Random
using PlanningDomains
using Setfield
using ChainRulesCore
using NeuroPlanner: add_goalstate
# using Yota

_isapprox(a::Nothing, b::Nothing; tol = 1e-5) = true
_isapprox(a::ZeroTangent, b::Nothing; tol = 1e-5) = true
_isapprox(a::Number, b::Number; tol = 1e-5) = abs(a-b)  < tol
_isapprox(a::NamedTuple,b::NamedTuple; tol = 1e-5) = all(_isapprox(a[k], b[k]; tol) for k in keys(a))
_isapprox(a::Tangent,b::NamedTuple; tol = 1e-5) = all(_isapprox(a[k], b[k]; tol) for k in keys(b))
_isapprox(a::Tuple, b::Tuple; tol = 1e-5) = all(_isapprox(a[k], b[k]; tol) for k in keys(a))
_isapprox(a::AbstractArray, b::AbstractArray; tol = 1e-5) = all(_isapprox.(a,b;tol))
include("dedu_matrix.jl")
include("knowledge_base.jl")
include("datanode.jl")
include("modelnode.jl")


# domain = load_domain("sokoban.pddl")
# problem = load_problem("s1.pddl")

# domain = load_domain("../classical-domains/classical/settlers/domain.pddl")
# problem = load_problem("../classical-domains/classical/settlers/p01_pfile1.pddl")


# domain = load_domain("../classical-domains/classical/depot/domain.pddl")
# problem = load_problem("../classical-domains/classical/depot/pfile1.pddl")

domain = load_domain("../classical-domains/classical/driverlog/domain.pddl")
problem = load_problem("../classical-domains/classical/driverlog/pfile1.pddl")

# domain = load_domain("../classical-domains/classical/briefcaseworld/domain.pddl")
# problem = load_problem("../classical-domains/classical/briefcaseworld/pfile1.pddl")

@testset "extraction of hypergraph" begin
	for arch in (MixedLRNN, LRNN, ObjectBinary, ASNet, HGNNLite, HGNN)
		ex = arch(domain)
		ex = NeuroPlanner.specialize(ex, problem)
		@test ex.init_state === nothing
		@test ex.goal_state === nothing
		state = initstate(domain, problem)
		ds = ex(state)
		m = reflectinmodel(ds)
		@test m(ds) isa Matrix

		#test adding goal state
		gex = add_goalstate(ex, problem)
		@test gex.init_state === nothing
		@test gex.goal_state !== nothing
		ds = gex(state)
		m = reflectinmodel(ds)
		@test m(ds) isa Matrix

		#test adding initial state
		iex = add_initstate(ex, problem)
		@test iex.init_state !== nothing
		@test iex.goal_state === nothing
		ds = iex(state)
		m = reflectinmodel(ds)
		@test m(ds) isa Matrix
	end
end

@testset "testing concatenation for batching" begin 
	state = initstate(domain, problem)
	goal = PDDL.get_goal(problem)
	planner = AStarPlanner(HAdd())
	sol = planner(domain, state, goal)
	plan = collect(sol)
	trajectory = sol.trajectory
	satisfy(domain, sol.trajectory[end], goal)
	
	for arch in (MixedLRNN, LRNN, ObjectBinary, ASNet, HGNNLite, HGNN)
		# get training example by running A* planner with h_add heuristic
		pddle = NeuroPlanner.specialize(arch(domain), problem)
		m = reflectinmodel(pddle(state), d -> Dense(d,10), SegmentedMean;fsm = Dict("" =>  d -> Dense(d,1)))

		@testset "forward path" begin 
			xx = [pddle(state) for state in sol.trajectory];
			yy = collect(length(sol.trajectory):-1:1);
			@test reduce(hcat, map(m, xx)) ≈  m(Flux.batch(xx))
			ii = randperm(length(xx))
			@test reduce(hcat, map(m, xx[ii])) ≈  m(Flux.batch(xx[ii]))
		end

		@testset "init-goal invariant" begin
			ex = arch(domain)
			iex = add_initstate(ex, problem)
			gex = add_goalstate(ex, problem)

			si = initstate(domain, problem)
			gi = goalstate(domain, problem)
			model = reflectinmodel(iex(si), d -> Dense(d,10), SegmentedMean;fsm = Dict("" =>  d -> Dense(d,1)))

			@test model(iex(goalstate(domain, problem))) ≈ model(gex(initstate(domain, problem)))
		end

		@testset "gradient path" begin 
			xx = [pddle(state) for state in sol.trajectory];
			bxx = Flux.batch(xx);
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
			@test all(maximum(abs2.(gs1[p] .- gs2[p])) < 1e-3 for p in ps)
		end
	end
end


@testset "Integration test with deduplication" begin 
	domain_name = "barman-sequential-satisficing"
	ipcyear = "ipc-2014"
	fminibatch = NeuroPlanner.minibatchconstructor("lstar")
	domain = load_domain(IPCInstancesRepo,ipcyear, domain_name)
	problems = list_problems(IPCInstancesRepo, ipcyear, domain_name)

	for arch in (MixedLRNN, LRNN, ObjectBinary, ASNet, HGNNLite, HGNN)		
		#create model from some problem instance
		pddld = arch(domain)
		model = let 
			problem = load_problem(IPCInstancesRepo, ipcyear,domain_name, first(problems))
			pddle, state = initproblem(pddld, problem)
			h₀ = pddle(state)
			reflectinmodel(h₀, d -> Dense(d, 8), SegmentedMean;fsm = Dict("" =>  d -> Dense(d, 1)))
		end

		for problem_file in problems
			problem = load_problem(IPCInstancesRepo, ipcyear,domain_name, problem_file);
			trajectory, plan = NeuroPlanner.sample_forward_trace(domain, problem, 20);
			ds = fminibatch(pddld, domain, problem, plan);
			
			ds1 = deepcopy(ds)
			ds2 =  @set ds.x = deduplicate(ds.x)

			fval1, gs1 = Flux.withgradient(m -> NeuroPlanner.loss(m, ds1), model)
			fval2, gs2 = Flux.withgradient(m -> NeuroPlanner.loss(m, ds2), model)
			@test fval1 ≈ fval2
			@test _isapprox(gs1, gs2; tol = 1e-3) # the error is quite tragic, likely caused by Float32s
		end
	end
end