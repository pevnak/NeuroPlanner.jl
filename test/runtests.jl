using NeuroPlanner
using PDDL
using Flux
using Mill
using GraphNeuralNetworks
using SymbolicPlanners
using Test
using Random
using NeuralAttentionlib

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


# pddle = PDDLExtractor(domain, problem) 


@testset "extraction and basic gradient of MultiModel" begin
	pddld = PDDLExtractor(domain) 
	pddle = NeuroPlanner.specialize(pddld, problem) 
	state = initstate(domain, problem)
	h₀ = pddle(state)
	m = MultiModel(h₀, 4, 1, d -> Dense(d,32))
	ps = Flux.params(m)
	gs = gradient(() -> sum(m(h₀)), ps)
	@test all(gs[p] !== nothing for p in ps)
end

@testset "extraction and basic gradient of hypergraph" begin
	ex = HyperExtractor(domain)
	ex = NeuroPlanner.specialize(ex, problem)
	gex = NeuroPlanner.add_goalstate(ex, problem)
	state = initstate(domain, problem)
	ds = ex(state)
	model = reflectinmodel(ds, d -> Dense(d,32), SegmentedMean)
	ps = Flux.params(model)
	gs = gradient(() -> sum(model(ds)), ps)
	@test all(gs[p] !== nothing for p in ps)

	# Let's toy with multihead attention
	x = model.im(ds.data)
    input_dims = size(x,1)
	head = 4
    head_dims = 4
    output_dims = 32

    mha = MultiheadAttention(head, input_dims, head_dims, output_dims)
   	model = (;mill = model, mha)
   	function f(model, ds) 
   		x = model.mill.im(ds.data)
   		x = model.mha(x)
   		model.mill.bm(model.mill.a(x, ds.bags))
   	end
	ps = Flux.params(model)
	gs = gradient(() -> sum(f(model, ds)), ps)
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

@testset "testing deduplication" begin 
    # @testset "deduplication test" begin
	domain = load_domain(domain_pddl)
	# pddld = PDDLExtractor(domain)
	pddld = HyperExtractor(domain)

	#create model from some problem instance
	model, dedup_model = let 
		problem = load_problem(first(problem_files))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		model = initmodel(h₀; graph_dim, graph_layers, dense_dim, dense_layers, heads, head_dims)
		dedup_model = reflectinmodel(h₀, d -> Dense(d,32), SegmentedMean)
		(model, dedup_model)
	end

 
	minibatches = map(train_files) do problem_file
		plan = deserialize(plan_file(problem_file))
		problem = load_problem(problem_file)
		ds = fminibatch(pddld, domain, problem, plan.plan)
		ds1 = deepcopy(ds)
		ds2 =  @set ds.x.data = deduplicate(dedup_model.im, ds.x.data)
		(;ds1, ds2)
	end

	m1 = deepcopy(model)
	m2 = deepcopy(model)
	ps1 = Flux.params(m1)
	ps2 = Flux.params(m2)
	opt1 = ADAM()
	opt2 = ADAM()
	debug = nothing
	@assert(all(p1 ≈ p2 for (p1,p2) in zip(ps1,ps2)))
	for (i, (ds1, ds2)) in enumerate(minibatches)
		l1, gs1 = withgradient(() -> NeuroPlanner.loss(m1,ds1), ps1)
		l2, gs2 = withgradient(() -> NeuroPlanner.loss(m2,ds2), ps2)
		Δdiff = sum([sum(abs2.(gs1[p1] .- gs2[p2])) for (p1,p2) in zip(ps1,ps2)])
		diff = sum([sum(abs2.(p1 .- p2)) for (p1,p2) in zip(ps1,ps2)])
		@show (diff, Δdiff)
		# if !all(gs1[p1] == gs2[p2] for (p1,p2) in zip(ps1,ps2))
		# 	debug = (ds1, ds2)
		# 	println(i," ",l1," ",l2)
		# 	break
		# end
		Flux.Optimise.update!(opt1, ps1, gs1)
		Flux.Optimise.update!(opt2, ps2, gs2)
	end

	map(train_files) do problem_file
		plan = deserialize(plan_file(problem_file))
		problem = load_problem(problem_file)
		ds1 = fminibatch(pddld, domain, problem, plan.plan)
		ds2 = @set ds1.x.data = deduplicate(model.mill.im, ds1.x.data)
		gs1 = fcollect(gradient(Base.Fix2(NeuroPlanner.loss, ds1), model))
		gs2 = fcollect(gradient(Base.Fix2(NeuroPlanner.loss, ds1), model))
		gs1 = filter(x -> x isa AbstractArray, gs1)
		gs2 = filter(x -> x isa AbstractArray, gs2)
		all(x ≈ y for (x,y) in zip(gs1, gs2))
	end

end