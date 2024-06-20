using NeuroPlanner
using NeuroPlanner.PDDL
using NeuroPlanner.Flux
using NeuroPlanner.Mill
using NeuroPlanner.SymbolicPlanners
using NeuroPlanner.Accessors
using NeuroPlanner.ChainRulesCore
using NeuroPlanner: add_goalstate
using Test
using Random
using PlanningDomains

ENCODINGS = (ObjectBinaryFE, ObjectBinaryFENA, ObjectBinaryME, ObjectAtom, ObjectAtomBipFE, 
			ObjectAtomBipFENA, ObjectAtomBipME, AtomBinaryFE, AtomBinaryFENA, AtomBinaryME, 
			ASNet, HGNNLite, HGNN)

@testset "extraction of hypergraph" begin
	@testset "Domain: $domain_name" for domain_name in DOMAINS
		domain, problem = load_problem_domain(domain_name) 
		@testset "architecture: $(arch) " for arch in ENCODINGS
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
end

@testset "testing concatenation for batching" begin 
	@testset "Domain: $domain_name" for domain_name in DOMAINS
		domain, problem = load_problem_domain(domain_name) 

		plan = load_plan(domain_name)
		state = initstate(domain, problem)
		trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
		# goal = PDDL.get_goal(problem)
		# Use the stuff below when you need to get a plan
		# planner = AStarPlanner(HAdd())
		# sol = planner(domain, state, goal)
		# plan = collect(sol)
		# save_plan(plan_path(domain_name), plan)
		# trajectory = sol.trajectory
		# satisfy(domain, sol.trajectory[end], goal)

		@testset "architecture: $(arch) " for arch in ENCODINGS
			# get training example by running A* planner with h_add heuristic
			pddle = NeuroPlanner.specialize(arch(domain), problem)
			m = reflectinmodel(pddle(state), d -> Dense(d,10), SegmentedMean;fsm = Dict("" =>  d -> Dense(d,1)))

			xx = [pddle(state) for state in trajectory];
			yy = collect(length(trajectory):-1:1);
			@test reduce(hcat, map(m, xx)) ≈  m(Flux.batch(xx))
			ii = randperm(length(xx))
			bxx = Flux.batch(xx[ii])
			ob = m(bxx)

			@testset "forward path" begin 
				@test reduce(hcat, map(m, xx[ii])) ≈  ob
			end

			@testset "forward path deduplication" begin 
				od = m(deduplicate(bxx))
				@test od ≈  ob
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

			# @testset "gradient path" begin 
			# 	xx = [pddle(state) for state in trajectory];
			# 	bxx = Flux.batch(xx);
			# 	yy = collect(length(trajectory):-1:1);

			# 	ps = Flux.params(m);
			# 	gs1 = gradient(ps) do 
			# 		map(xx, yy) do h₀, y
			# 			(sum(m(h₀)) - y)^2
			# 		end |> sum 
			# 	end;

			# 	gs2 = gradient(ps) do 
			# 		sum((vec(m(bxx)) .- yy) .^ 2)
			# 	end;
			# 	@test _isapprox(gs1, gs2; atol = 1e-3)
			# end
		end
	end
end


######
#	Super-expensive integration test
######

# @testset "Integration test with deduplication" begin 
# 	@testset "Domain: $domain_name" for domain_name in DOMAINS
# 		domain, problem = load_problem_domain(domain_name) 
# 		plan = load_plan(domain_name)
# 		state = initstate(domain, problem)
# 		trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)

# 		for arch in ENCODINGS
# 			#create model from some problem instance
# 			pddld = arch(domain)
# 			model = let 
# 				pddle, state = initproblem(pddld, problem)
# 				h₀ = pddle(state)
# 				reflectinmodel(h₀, d -> Dense(d, 8), SegmentedMean;fsm = Dict("" =>  d -> Dense(d, 1)))
# 			end

# 			@testset "Loss: $(loss)" for loss in ["lstar","l2","lrt","lbfs","bellman"]
# 				fminibatch = NeuroPlanner.minibatchconstructor("lstar")

# 				ds = fminibatch(pddld, domain, problem, plan);			
# 				ds1 = deepcopy(ds)
# 				ds2 =  @set ds.x = deduplicate(ds.x)

# 				fval1 = model(ds1.x)
# 				fval2 = model(ds2.x)
# 				@test fval1 ≈ fval2

# 				fval1 = NeuroPlanner.loss(model, ds1)
# 				fval2 = NeuroPlanner.loss(model, ds2)
# 				@test fval1 ≈ fval2
# 				# fval1, gs1 = Flux.withgradient(m -> NeuroPlanner.loss(m, ds1), model)
# 				# fval2, gs2 = Flux.withgradient(m -> NeuroPlanner.loss(m, ds2), model)
# 				# @test fval1 ≈ fval2
# 				# @test _isapprox(gs1, gs2; atol = 1e-3) # the error is quite tragic, likely caused by Float32s
# 			end
# 		end
# 	end
# end


# @testset "Integration test with deduplication" begin 
# 	domain_name = "barman-sequential-satisficing"
# 	ipcyear = "ipc-2014"
# 	fminibatch = NeuroPlanner.minibatchconstructor("lstar")
# 	domain = load_domain(IPCInstancesRepo,ipcyear, domain_name)
# 	problems = list_problems(IPCInstancesRepo, ipcyear, domain_name)

# 	for arch in ENCODINGS
# 		#create model from some problem instance
# 		pddld = arch(domain)
# 		model = let 
# 			problem = load_problem(IPCInstancesRepo, ipcyear,domain_name, first(problems))
# 			pddle, state = initproblem(pddld, problem)
# 			h₀ = pddle(state)
# 			reflectinmodel(h₀, d -> Dense(d, 8), SegmentedMean;fsm = Dict("" =>  d -> Dense(d, 1)))
# 		end

# 		for problem_file in problems
# 			problem = load_problem(IPCInstancesRepo, ipcyear,domain_name, problem_file);
# 			trajectory, plan = NeuroPlanner.sample_forward_trace(domain, problem, 20);
# 			ds = fminibatch(pddld, domain, problem, plan);
			
# 			ds1 = deepcopy(ds)
# 			ds2 =  @set ds.x = deduplicate(ds.x)

# 			fval1, gs1 = Flux.withgradient(m -> NeuroPlanner.loss(m, ds1), model)
# 			fval2, gs2 = Flux.withgradient(m -> NeuroPlanner.loss(m, ds2), model)
# 			@test fval1 ≈ fval2
# 			@test _isapprox(gs1, gs2; tol = 1e-3) # the error is quite tragic, likely caused by Float32s
# 		end
# 	end
# end