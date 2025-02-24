

########
#	BellmanLoss
########
"""
A bellman loss function combining inequalities in the expanded actions with L2 loss 
as proposed in 
Ståhlberg, Simon, Blai Bonet, and Hector Geffner. "Learning Generalized Policies without Supervision Using GNNs.", 2022
Equation (15)
"""
struct BellmanMiniBatch{X,Y} <: AbstractMinibatch
	x::X 
	path_cost::Y
	trajectory_states::Vector{Int}
	child_states::Vector{Vector{Int}}
	cost_to_goal::Vector{Float32}
	sol_length::Int64
end

function BellmanMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:GenericState}; goal_aware = true, goal_state = goalstate(pddld.domain, problem), max_branch = typemax(Int))
	pddle = goal_aware ? add_goalstate(pddld, problem, goal_state) : specialize(pddld, problem)
	state = trajectory[1]
	spec = Specification(problem)

	stateids = Dict(hash(state) => 1)
	states = [(g = 0, state = state)]
	trajectory_states = Vector{Int}()
	child_states = Vector{Vector{Int}}()

	htrajectory = hash.(trajectory)
	for i in 1:(length(trajectory)-1)
		sᵢ, sⱼ = trajectory[i], trajectory[i+1]
		hsⱼ = hash(sⱼ)
		gᵢ = states[stateids[hash(sᵢ)]].g
		next_states = _next_states(domain, problem, sᵢ, st)
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
		@assert hash(sⱼ) ∈ keys(stateids) "next state on the trajectory is not in the oppen list"

		push!(child_states, [stateids[x.id] for x in next_states])
	end
	path_cost = [s.g for s in states]
	x = batch([pddle(s.state) for s in states])
	trajectory_states = [stateids[hash(s)] for s in trajectory]
	cost_to_goal = Float32.(collect(length(trajectory):-1:1))
	BellmanMiniBatch(x, path_cost, trajectory_states, child_states, cost_to_goal, length(trajectory))
end

function BellmanMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, trajectory::AbstractVector{<:GenericState}; kwargs...)
	BellmanMiniBatch(pddld, domain, problem, nothing, trajectory; kwargs...)
end

function BellmanMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; kwargs...)
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	BellmanMiniBatch(pddld, domain, problem, trajectory; kwargs...)
end


function bellmanloss(model, x, g, trajectory_states, child_states, cost_to_goal, surrogate=softplus)
	f = model(x)
	v = f[trajectory_states]
	vs = cost_to_goal
	reg = mean(max.(0, vs - v) + max.(0, v - 2*vs))
	isempty(child_states) && return(reg)
	h = map(1:length(child_states)) do i 
		max(1 + minimum(f[child_states[i]]) - f[trajectory_states[i]], 0)
	end
	mean(h) + reg 
end

bellmanloss(model, xy::BellmanMiniBatch, surrogate = softplus) = bellmanloss(model, xy.x, xy.path_cost, xy.trajectory_states, xy.child_states, xy.cost_to_goal)
