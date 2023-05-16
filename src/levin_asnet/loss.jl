#############
#	L2 Losses
#############
struct LevinMiniBatch{X,Y, P}
	x::X 
	y::Y
	policy::P
end


function get_action_id(ex::LevinASNet, a)
	assignment = Dict(zip(ex.domain.actions[a.name].args, a.args))
	predicates = NeuroPlanner.extract_predicates(ex.domain.actions[a.name])
	k = (a.name, [NeuroPlanner.ground(p, assignment) for p in predicates]...)
	ex.action2id[k]
end

function LevinMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; goal_aware = true, max_branch = typemax(Int))
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	LevinMiniBatch(pddld, domain, problem, plan, trajectory; goal_aware, max_branch)
end

function LevinMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int))
	pddle = isspecialized(pddld) ? pddld : specialize(pddld, problem)
	pddle = goal_aware ? add_goalstate(pddle, problem) : pddle

	x = batch(map(pddle, trajectory))
    y = collect(length(trajectory):-1:1)
    policy = Flux.onehotbatch(map(a -> get_action_id(pddle, a), plan), 1:length(pddle.action2id))
	LevinMiniBatch(x, y, policy)
end

function loss(model, xy::LevinMiniBatch, surrogate=softplus) 
	heuristic, policy = model(xy.x)
	isempty(xy.policy) && return(Flux.Losses.mse(vec(heuristic), xy.y))
	Flux.Losses.mse(vec(heuristic), xy.y) + Flux.Losses.logitcrossentropy(policy[:, 1:end-1], xy.policy)
end