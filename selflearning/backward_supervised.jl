using NeuroPlanner
using PDDL
using Flux
using JSON
using SymbolicPlanners
using PDDL: GenericProblem
using SymbolicPlanners: PathSearchSolution
using Statistics
using IterTools
using Random
using StatsBase
using Serialization
using DataFrames
using Mill
using Functors
using Accessors
using Logging
using TensorBoardLogger

include("solution_tracking.jl")
include("problems.jl")
include("training.jl")
include("utils.jl")

function experiment(domain_name, hnet, domain_pddl, train_files, problem_files, filename, fminibatch;max_steps = 10000, max_time = 30, graph_layers = 2, residual = true, dense_layers = 2, dense_dim = 32, settings = nothing)
	!isdir(dirname(filename)) && mkpath(dirname(filename))
	domain = load_domain(domain_pddl)
	pddld = hnet(domain; message_passes = graph_layers, residual)

	#create model from some problem instance

	# we can check that the model can learn the training data if the dedup_model can differentiate all input states, which is interesting by no means

	model = if isfile(filename*"_model.jls")
		deserialize(filename*"_model.jls")
	else
		model = let 
			problem = load_problem(first(train_files))
			pddle, state = initproblem(pddld, problem)
			h₀ = pddle(state)
			# model = reflectinmodel(h₀, d -> ffnn(d, dense_dim, dense_dim, dense_layers);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
			reflectinmodel(h₀, d -> Dense(d, dense_dim, relu);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
		end

		# logger=tblogger(filename*"_events.pb")
		logger=nothing
		t = @elapsed minibatches = map(train_files) do problem_file
			@show problem_file
			println("creating sample from problem: ",problem_file)
			plan = load_plan(problem_file)
			problem = load_problem(problem_file)
			ds = fminibatch(pddld, domain, problem, plan)
			if fminibatch == NeuroPlanner.BidirecationalLₛMiniBatch
				dedu = [(@set s.x = deduplicate(s.x)) for s in ds]
			else
				dedu = @set ds.x = deduplicate(ds.x)
			end
			size_o, size_d =  Base.summarysize(ds), Base.summarysize(dedu)
			println("original: ", size_o, " dedupped: ", size_d, " (",round(100*size_d / size_o, digits =2),"%)")
			dedu
		end
		if fminibatch == NeuroPlanner.BidirecationalLₛMiniBatch
			minibatches = reduce(vcat, minibatches)
		end
		# log_value(logger, "time_minibatch", t; step=0)
		opt = AdaBelief();
		ps = Flux.params(model);
		t = @elapsed train!(NeuroPlanner.loss, model, ps, opt, () -> rand(minibatches), max_steps; logger, trn_data = minibatches)
		# log_value(logger, "time_train", t; step=0)
		serialize(filename*"_model.jls", model)	
		model
	end

	# BiGreedyPlanner is not yet working because of some small bug
	planners = model isa NeuroPlanner.LevinModel ? [BFSPlanner] : [AStarPlanner, GreedyPlanner, BackwardAStarPlanner, BackwardGreedyPlanner, BiAStarPlanner]

	stats = map(Iterators.product(planners, problem_files)) do (planner, problem_file)
		used_in_train = problem_file ∈ train_files
		@show problem_file
		sol = solve_problem(pddld, problem_file, model, planner; return_unsolved = true)
		trajectory = sol.sol.status == :max_time ? nothing : sol.sol.trajectory
		merge(sol.stats, (;used_in_train, planner = "$(planner)", trajectory, problem_file))
	end
	df = DataFrame(vec(stats))
	mean(df.solved[.!df.used_in_train])
	serialize(filename*"_stats.jls", stats)
	settings !== nothing && serialize(filename*"_settings.jls",settings)
end

"""
ArgParse example implemented in Comonicon.

# Arguments

- `problem_name`: a name of the problem to solve ("ferry", "gripper", "blocks", "npuzzle", "elevator_00")
- `arch_name`: an architecture of the neural network implementing heuristic("asnet", "pddl")
- `loss_name`: 

# Options

- `--max_steps <Int>`: maximum number of steps of SGD algorithm (default 10_000)
- `--max_time <Int>`:  maximum steps of the planner used for evaluation (default 30)
- `--graph_layers <Int>`:  maximum number of layers of (H)GNN (default 1)
- `--dense_dim <Int>`:  dimension of all hidden layers, even those realizing graph convolutions (default  32)
- `--dense_layers <Int>`:  number of layers of dense network after pooling vertices (default 32)
- `--residual <String>`:  residual connections between graph convolutions (none / dense / linear)

max_steps = 10_000; max_time = 30; graph_layers = 2; dense_dim = 16; dense_layers = 2; residual = "none"; seed = 1
domain_name = "ferry"
loss_name = "backlstar"
arch_name = "pddl"
"""

@main function main(domain_name, arch_name, loss_name; max_steps::Int = 10_000, max_time::Int = 30, graph_layers::Int = 1, 
		dense_dim::Int = 32, dense_layers::Int = 2, residual::String = "none", seed::Int = 1)
	Random.seed!(seed)
	settings = (;domain_name, arch_name, loss_name, max_steps, max_time, graph_layers, dense_dim, dense_layers, residual, seed)
	@show settings
	archs = Dict("asnet" => ASNet, "pddl" => HyperExtractor, "hgnnlite" => HGNNLite, "hgnn" => HGNN, "levinasnet" => LevinASNet)
	residual = Symbol(residual)
	domain_pddl, problem_files = getproblem(domain_name, false)
	# problem_files = filter(s -> isfile(plan_file(domain_name, s)), problem_files)
	train_files = filter(s -> isfile(plan_file(s)), problem_files)
	train_files = sample(train_files, min(div(length(problem_files), 2), length(train_files)), replace = false)
	fminibatch = NeuroPlanner.minibatchconstructor(loss_name)
	hnet = archs[arch_name]

	filename = joinpath("bidirectional", domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_"))
	experiment(domain_name, hnet, domain_pddl, train_files, problem_files, filename, fminibatch; max_steps, max_time, graph_layers, residual, dense_layers, dense_dim, settings)
end

function debug()
	# assuming we have a trained model, let's take a look on loss function
	df = map(zip(minibatches, train_files)) do (mb, tf )
		(;tf = basename(tf), loss = NeuroPlanner.loss(model, mb, x -> x > 0))
	end |> DataFrame
	# This tells us that we are making few errors, not sure, if this is important

	#Let's try to solve the problem
	plan = load_plan(problem_file)
	problem = load_problem(problem_file)
	back_sol = solve_problem(pddld, problem_file, model, BackwardAStarPlanner; return_unsolved = false)
	# even if we have a small error, the problem seems to be unresolved. 


	hfun = NeuroHeuristic(pddld, problem, model; backward = true)
	planner = BackwardAStarPlanner(hfun; max_time, save_search = true)
	sol = planner(domain, problem)

	# Let's see, how many states from the plan are in the search-tree
	bt = NeuroPlanner.backward_simulate(domain, problem, plan)
	ft = SymbolicPlanners.simulate(StateRecorder(), domain, initstate(domain, problem), plan)
	st = sol.search_tree
	map(s -> any(issubset(b.state,s) for b in values(st)), bt)

	s = goalstate(domain, problem)
	for sol_a in reverse(plan)
		hs = map(relevant(domain, s)) do a
			s₋₁ = PDDL.regress(domain, s, a)
			h = hfun(s₋₁)
			(h, a == sol_a)
		end

		if argmin(first, hs)[2]
			for (h, a) in hs
				a ? print(@yellow "$(h) ") : print(h, " ")
			end
		else
			for (h, a) in hs
				a ? print(@red "$(h) ") : print(h, " ")
			end
		end
		println()
		s = PDDL.regress(domain, s, sol_a)
	end

	# what is the average branching factor of the backward search?
	g = goalstate(domain, problem)
	i = initstate(domain, problem)
	plan = load_plan(problem_file)
	consts = Symbol.(("not-eq", "car","location",))
	additional = filter(s -> s.name ∈ consts, i.facts)
	foreach(s -> push!(g.facts, s), additional)

	let s = goalstate(domain, problem)
		map(reverse(plan)) do a 
			n = length(relevant(domain, s))
			s = PDDL.regress(domain, s, a)
			n
		end
	end

	let s = initstate(domain, problem)
		map(plan) do a 
			n = length(available(domain, s))
			s = PDDL.transition(domain, s, a)
			n
		end
	end

end


function understanding_bidirectional_search()
	plan = load_plan(problem_file)
	problem = load_problem(problem_file)

	back_sol = solve_problem(pddld, problem_file, model, BackwardAStarPlanner; return_unsolved = false)
	forw_sol = solve_problem(pddld, problem_file, model, AStarPlanner; return_unsolved = false)

	bi_sol = solve_problem(pddld, problem_file, model, BiAStarPlanner; return_unsolved = false)

	bst = bi_sol.sol.b_search_tree
	fst = bi_sol.sol.f_search_tree
	plan = bi_sol.sol.plan

	# I want to verify, if both have followed the same route
	bt = NeuroPlanner.backward_simulate(domain, problem, plan)
	ft = SymbolicPlanners.simulate(StateRecorder(), domain, initstate(domain, problem), plan)

	# Let's check that states from backward replay are subset of states on the forward path.
	# This should hold, but I want to to double check
	map(zip(ft, reverse(bt))) do (f, b )
		issubset(b, f)
	end

	# Let's now check how many of states on the optimal reverse path are in the search tree
	[haskey(bst, hash(s)) for s in bt]
	[haskey(fst, hash(s)) for s in ft]



	s = goalstate(domain, problem)
	for sol_a in reverse(plan)
		hs = map(relevant(domain, s)) do a
			s₋₁ = PDDL.regress(domain, s, a)
			h = only(hfun.model(hfun.pddle(s₋₁)))
			(h, a == sol_a)
		end

		if argmin(first, hs)[2]
			for (h, a) in hs
				a ? print(@yellow "$(h) ") : print(h, " ")
			end
		else
			for (h, a) in hs
				a ? print(@red "$(h) ") : print(h, " ")
			end
		end
		println()
		s = PDDL.regress(domain, s, sol_a)
	end

	only(hfun.model(hfun.pddle(goalstate(domain, problem))))


	# Let's run the planner
	hfun = EvalTracker(NeuroHeuristic(NeuroPlanner.add_initstate(pddld, problem), model, Ref(0.0)))
	planner = BackwardAStarPlanner(hfun; max_time = 30, save_search = true)
	sol = planner(domain, problem)
	plot_search_tree("/tmp/backward", sol, hfun)
end

