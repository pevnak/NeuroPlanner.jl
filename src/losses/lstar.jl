#############
#	Lstar Losses as described in  Chrestien, Leah, et al. "Optimize planning heuristics to rank, not to estimate cost-to-goal." Advances in Neural Information Processing Systems 36 (2024).
#############
struct LₛMiniBatch{X,H,Y}
	x::X 
	H₊::H 
	H₋::H 
	path_cost::Y
	sol_length::Int64
end

struct UnsolvedLₛ{S,D,P}
	sol::S 
	pddld::D
	problem::P 
end


function LₛMiniBatch(sol, pddld, problem::GenericProblem)
	sol.search_tree === nothing && error("solve the problem with `save_search=true` to keep the search tree")
	sol.status != :success && return(UnsolvedLₛ(sol, pddld, problem))
	pddle = NeuroPlanner.initproblem(pddld, problem)[1]
	trajectory = sol.trajectory
	LₛMiniBatch(sol, pddle, trajectory)
end

function prepare_minibatch(mb::UnsolvedLₛ)
	@unpack sol, problem, pddld = mb
	trajectory = artificial_trajectory(sol)
	goal = trajectory[end]
	pddle = NeuroPlanner.add_goalstate(pddld, problem, goal)
	LₛMiniBatch(sol, pddle, trajectory)
end

function LₛMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; goal_aware = true, max_branch = typemax(Int), plot_dict=nothing)
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	LₛMiniBatch(pddld, domain, problem, trajectory; goal_aware, max_branch, plot_dict)
end

function LₛMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int), plot_dict=nothing)
	LₛMiniBatch(pddld, domain, problem, nothing, trajectory; goal_aware, max_branch, plot_dict)
end

_next_states(domain, problem, sᵢ, st::RSearchTree) = st.st[hash(sᵢ)]
function _next_states(domain, problem, sᵢ, st::Nothing) 
	acts = available(domain, sᵢ)
	isempty(acts) && return([])
	hsᵢ = hash(sᵢ)
	map(acts) do act
		state = PDDL.execute(domain, sᵢ, act; check=false)
		(;state,
		  id = hash(state),
		  parent_action = act,
		  parent_id = hsᵢ
		 )
	end
end

function LₛMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int), plot_dict=nothing)
	pddle = goal_aware ? NeuroPlanner.add_goalstate(pddld, problem) : specialize(pddld, problem)
	state = trajectory[1]
	spec = Specification(problem)

	stateids = Dict(hash(state) => 1)
	states = [(g = 0, state = state)]
	I₊ = Vector{Int64}()
	I₋ = Vector{Int64}()

	htrajectory = hash.(trajectory)
	for i in 1:(length(trajectory)-1)
		if !isnothing(plot_dict)
			depth_indexs = get!(Vector{Int}, plot_dict, :depth)
			push!(depth_indexs, length(stateids))
		end

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
			if next_state.state ∈ trajectory && !isnothing(plot_dict)
				traj_indexes = get!(Vector{Int}, plot_dict, :traj)
				push!(traj_indexes, stateids[next_state.id])
			end
		end
		@assert hash(sⱼ) ∈ keys(stateids)
		open_set = setdiff(keys(stateids), htrajectory)

		for s in open_set
			#TODO sem dat nerovnost
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
	inner_type = typeof(pddle(first(states).state))
	x = batch(inner_type[pddle(s.state) for s in states])
	LₛMiniBatch(x, H₊, H₋, path_cost, length(trajectory))
end


function lₛloss(model, x, g, H₊, H₋, surrogate=softplus)
	g = reshape(g, 1, :)
	f = model(x) + g
	o = f * H₋ - f * H₊
	isempty(o) && return(zero(eltype(o)))
	mean(surrogate.(o))
end

lₛloss(model, xy::LₛMiniBatch) = lₛloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋)
lₛloss(model, xy::LₛMiniBatch, surrogate) = lₛloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋, surrogate)
lₛloss(model, mb::NamedTuple{(:minibatch,:stats)}) = lₛloss(model, mb.minibatch)
