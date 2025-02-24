
struct L₂MiniBatch{X,Y}
	x::X 
	y::Y
end

struct UnsolvedL₂{S,D,P}
	sol::S 
	pddld::D
	problem::P 
end

function L₂MiniBatch(pddle, trajectory::AbstractVector{<:GenericState})
	L₂MiniBatch(batch(map(pddle, trajectory)),
     collect(length(trajectory):-1:1),
     )
end


function L₂MiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int))
	pddle = goal_aware ? add_goalstate(pddld, problem) : pddld
	L₂MiniBatch(pddle, trajectory)
end

function L₂MiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int))
	L₂MiniBatch(pddld, domain, problem, trajectory; goal_aware, max_branch)
end

function L₂MiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; goal_aware = true, max_branch = typemax(Int))
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	L₂MiniBatch(pddld, domain, problem, trajectory; goal_aware, max_branch)
end


function prepare_minibatch(mb::UnsolvedL₂)
	@unpack sol, problem, pddld = mb
	trajectory = artificial_trajectory(sol)
	goal = trajectory[end]
	pddle = NeuroPlanner.add_goalstate(pddld, problem, goal)
	L₂MiniBatch(sol, pddle, trajectory)
end 

struct L₂Loss end 
l₂loss(model, x, y) = Flux.Losses.mse(vec(model(x)), y)
l₂loss(model, xy::L₂MiniBatch) = l₂loss(model, xy.x, xy.y)
l₂loss(model, xy::L₂MiniBatch, surrogate) = l₂loss(model, xy.x, xy.y)
l₂loss(model, mb::NamedTuple{(:minibatch,:stats)}) = l₂loss(model, mb.minibatch)
(l::L₂Loss)(args...) = l₂loss(args...)
