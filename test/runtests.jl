using NeuroPlanner
using PDDL
using Flux
using Mill
using SymbolicPlanners
using Test
using Random
using PlanningDomains
using Setfield

include("knowledge_base.jl")

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

@testset "extraction and basic gradient of hypergraph" begin
	ex = HyperExtractor(domain)
	ex = ASNet(domain)
	ex = HGNNLite(domain)
	ex = HGNN(domain)
	ex = NeuroPlanner.specialize(ex, problem)
	gex = NeuroPlanner.add_goalstate(ex, problem)
	state = initstate(domain, problem)
	ds = ex(state)
	m = reflectinmodel(ds)
	m(ds)

	ds = gex(state)
	m = reflectinmodel(ds)
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

@testset "Integration test without deduplication" begin 
	domain_name = "barman-sequential-satisficing"
	ipcyear = "ipc-2014"
	fminibatch = NeuroPlanner.minibatchconstructor("lstar")
	domain = load_domain(IPCInstancesRepo,ipcyear, domain_name)
	pddld = HyperExtractor(domain; message_passes = 1, residual = :none)
	problems = list_problems(IPCInstancesRepo, ipcyear, domain_name)


	#create model from some problem instance
	model = let 
		problem = load_problem(IPCInstancesRepo, ipcyear,domain_name, first(problems))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		reflectinmodel(h₀, d -> Dense(d, 8), SegmentedSum;fsm = Dict("" =>  d -> Dense(d, 1)))
	end

	ps = Flux.params(model);
	for problem_file in problems
		@show problem_file
		problem = load_problem(IPCInstancesRepo, ipcyear,domain_name, problem_file)
		trajectory, plan = NeuroPlanner.sample_forward_trace(domain, problem, 20)
		ds = fminibatch(pddld, domain, problem, plan)
		fval, gs = Flux.withgradient(() -> NeuroPlanner.loss(model, ds), ps)
		@test any(sum(abs.(gs[p])) > 0 for p in ps)
	end
end

@testset "Integration test with deduplication" begin 
	domain_name = "barman-sequential-satisficing"
	ipcyear = "ipc-2014"
	fminibatch = NeuroPlanner.minibatchconstructor("lstar")
	domain = load_domain(IPCInstancesRepo,ipcyear, domain_name)
	pddld = HyperExtractor(domain)
	problems = list_problems(IPCInstancesRepo, ipcyear, domain_name)


	#create model from some problem instance
	model, dedup_model = let 
		problem = load_problem(IPCInstancesRepo, ipcyear,domain_name, first(problems))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		model = reflectinmodel(h₀, d -> Chain(Dense(d,32,relu),Dense(32,32));fsm = Dict("" =>  d -> Chain(Dense(d,32,relu),Dense(32,1))))
		dedup_model = reflectinmodel(h₀, d -> Dense(d,32) ;fsm = Dict("" =>  d -> Dense(d, 32)))
		(model, dedup_model)
	end

	ps = Flux.params(model);
	for problem_file in problems
		problem = load_problem(IPCInstancesRepo, ipcyear,domain_name, problem_file);
		trajectory, plan = NeuroPlanner.sample_forward_trace(domain, problem, 20);
		ds = fminibatch(pddld, domain, problem, plan);
		
		ds1 = deepcopy(ds)
		ds2 =  @set ds.x = deduplicate(dedup_model, ds.x)

		fval1, gs1 = Flux.withgradient(() -> NeuroPlanner.loss(model, ds1), ps)
		fval2, gs2 = Flux.withgradient(() -> NeuroPlanner.loss(model, ds2), ps)
		@test fval1 ≈ fval2
		sum([sum(abs2.(gs1[p] .- gs2[p])) for p in ps])


		fval1, gs1 = Flux.withgradient(() -> sum(_apply_layers(ds1.x, model)[:x2]), ps)
		fval2, gs2 = Flux.withgradient(() -> sum(_apply_layers(ds2.x, model)[:x2]), ps)
		fval1, gs1 = Flux.withgradient(() -> sum(model[:x2][:contains](ds1.x, ds1.x[:x2][:contains])), ps)
		fval2, gs2 = Flux.withgradient(() -> sum(model[:x2][:contains](ds2.x, ds2.x[:x2][:contains])), ps)
		@test fval1 ≈ fval2
		@test all([sum(abs2.(gs1[p] .- gs2[p])) < 1e-6 for p in ps if gs1[p] !== nothing])

		# kb1 = ds1.x 
		# kb2 = ds2.x 
		# k = :contains
		# _m = model[:x2][k]
		# _ds1 = ds1.x[:x2][k]
		# _ds2 = ds2.x[:x2][k]

		# _dss1 = BagNode(ProductNode((Matrix(kb1, _ds1.data.data[1].data), Matrix(kb1, _ds1.data.data[2].data))), _ds1.bags)
		# _dss2 = BagNode(ProductNode((Matrix(kb1, _ds2.data.data[1].data), Matrix(kb1, _ds2.data.data[2].data))), _ds2.bags)

		# fval1, gs1 = Flux.withgradient(() -> sum(_m(_dss1)), ps)
		# fval2, gs2 = Flux.withgradient(() -> sum(_m(_dss2)), ps)
		# @test fval1 ≈ fval2
		# sum([sum(abs2.(gs1[p] .- gs2[p])) for p in ps if gs1[p] !== nothing])

		# ds1 = BagNode(ArrayNode([1 2 1; 1 2 1;]), [1:3])
		# ds2 = BagNode(ArrayNode([1 2; 1 2]), ScatteredBags([[1,2,1]]))
		# m = BagModel(ArrayModel(Dense(2,2)), SegmentedSum(2), identity)
		# ps = Flux.params(m)
		# fval1, gs1 = Flux.withgradient(() -> sum(m(ds1)), ps)
		# fval2, gs2 = Flux.withgradient(() -> sum(m(ds2)), ps)
		# @test fval1 ≈ fval2
		# sum([sum(abs2.(gs1[p] .- gs2[p])) for p in ps if gs1[p] !== nothing])

	end

	###########
	#	The commented part below tracks how the gradients starts to slowly diverge
	###########
	# minibatches = map(problems) do problem_file
	# 	problem = load_problem(IPCInstancesRepo, ipcyear,domain_name, problem_file)
	# 	trajectory, plan = NeuroPlanner.sample_forward_trace(domain, problem, 20)
	# 	ds = fminibatch(pddld, domain, problem, plan)
	# 	ds1 = deepcopy(ds)
	# 	ds2 =  @set ds.x = deduplicate(dedup_model, ds.x)
	# 	(;ds1, ds2)
	# end
	# m1 = deepcopy(model)
	# m2 = deepcopy(model)
	# ps1 = Flux.params(m1)
	# ps2 = Flux.params(m2)
	# opt1 = ADAM()
	# opt2 = ADAM()
	# debug = nothing
	# @assert(all(p1 ≈ p2 for (p1,p2) in zip(ps1,ps2)))
	# for (i, (ds1, ds2)) in enumerate(minibatches)
	# 	l1, gs1 = withgradient(() -> NeuroPlanner.loss(m1,ds1), ps1)
	# 	l2, gs2 = withgradient(() -> NeuroPlanner.loss(m2,ds2), ps2)
	# 	Δdiff = sum([sum(abs2.(gs1[p1] .- gs2[p2])) for (p1,p2) in zip(ps1,ps2)])
	# 	diff = sum([sum(abs2.(p1 .- p2)) for (p1,p2) in zip(ps1,ps2)])
	# 	@show (diff, Δdiff)
	# 	# if !all(gs1[p1] == gs2[p2] for (p1,p2) in zip(ps1,ps2))
	# 	# 	debug = (ds1, ds2)
	# 	# 	println(i," ",l1," ",l2)
	# 	# 	break
	# 	# end
	# 	Flux.Optimise.update!(opt1, ps1, gs1)
	# 	Flux.Optimise.update!(opt2, ps2, gs2)
	# end
end