function _previous_states(domain, problem, sᵢ) 
	acts = relevant(domain, sᵢ)
	isempty(acts) && return([])
	hsᵢ = hash(sᵢ)
	map(acts) do act
		state = PDDL.regress(domain, sᵢ, act; check=false)
		(;state,
		  id = hash(state),
		  parent_action = act,
		  parent_id = hsᵢ
		 )
	end
end

function backward_simulate(domain, problem, plan)
	s₀ = goalstate(domain, problem)
	trajectory = accumulate(reverse(plan), init = s₀) do s,a
		PDDL.regress(domain, s, a)
	end
	vcat([s₀], trajectory)
end

function BackwardL₂MiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Compound}; goal_aware = true, max_branch = typemax(Int))
	pddle = goal_aware ? NeuroPlanner.add_initstate(pddld, problem) : pddld
	trajectory = backward_simulate(domain, problem, plan)
	L₂MiniBatch(batch(map(pddle, trajectory)),
     collect(length(trajectory):-1:1),
     )
end

function BackwardLₛMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Compound}; goal_aware = true, max_branch = typemax(Int))
	pddle = goal_aware ? NeuroPlanner.add_initstate(pddld, problem) : pddld
	trajectory = backward_simulate(domain, problem, plan)
	plan = reverse(plan)
	spec = Specification(problem)
	state = trajectory[1]
	stateids = Dict(hash(state) => 1)
	states = [(g = 0, state = state)]
	I₊ = Vector{Int64}()
	I₋ = Vector{Int64}()

	htrajectory = hash.(trajectory)
	for i in 1:(length(trajectory)-1)
		sᵢ, sⱼ = trajectory[i], trajectory[i+1]
		hsⱼ = hash(sⱼ)
		gᵢ = states[stateids[hash(sᵢ)]].g
		next_states = _previous_states(domain, problem, sᵢ)
		if length(next_states) > max_branch # limit the branching factor is not excessive
			ii = findall(s -> s.state == sⱼ, next_states)
			ii = union(ii, sample(1:length(next_states), max_branch, replace = false))
			next_states = next_states[ii]
		end
		isempty(next_states) && error("the inner node is not in the search tree")
		for next_state in next_states
			act = next_state.parent_action
			act_cost = get_cost(spec, domain, sᵢ, act, next_state.state)
			next_state.id ∈ keys(stateids) && continue
			stateids[next_state.id] = length(stateids) + 1
			push!(states, (;g = gᵢ + act_cost, state = next_state.state))
		end
		@assert hash(sⱼ) ∈ keys(stateids)
		open_set = setdiff(keys(stateids), htrajectory)

		for s in open_set
			push!(I₊, stateids[s])
			push!(I₋, stateids[hsⱼ])
		end
	end
	if isempty(I₊)
		H₊ = onehotbatch([], 1:length(stateids))
		H₋ = onehotbatch([], 1:length(stateids))
	else
		H₊ = onehotbatch(I₊, 1:length(stateids))
		H₋ = onehotbatch(I₋, 1:length(stateids))
	end
	path_cost = [s.g for s in states]
	x = batch([pddle(s.state) for s in states])
	LₛMiniBatch(x, H₊, H₋, path_cost, length(trajectory))
end


function BackwardLgbfsMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Compound}; kwargs...)
	l = BackwardLₛMiniBatch(pddld, domain, problem, plan; kwargs...)
	LgbfsMiniBatch(l.x, l.H₊, l.H₋, l.path_cost, l.sol_length)	
end

function BackwardLRTMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Compound}; goal_aware = true, kwargs...)
	pddle = goal_aware ? NeuroPlanner.add_initstate(pddld, problem) : pddld
	trajectory = backward_simulate(domain, problem, plan)
	n = length(trajectory)
	if n < 2
		H₊ = onehotbatch([], 1:n)
		H₋ = onehotbatch([], 1:n)
	else
		H₊ = onehotbatch(1:n -1 , 1:n)
		H₋ = onehotbatch(2:n, 1:n)
	end
	LRTMiniBatch(batch(map(pddle, trajectory)),
		H₊,
		H₋,
		collect(length(trajectory):-1:1),
		n
    )
end


