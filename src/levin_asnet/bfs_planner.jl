SymbolicPlanners.@auto_hash_equals mutable struct BFSPathNode{S<:State}
    id::UInt
    state::S
    path_cost::Float32
    log_path_prob::Float32
    log_action_probs::Vector{Float32}
    parent_id::Union{UInt,Nothing}
    parent_action::Union{Term,Nothing}
end

BFSPathNode(id, state::S, path_cost, log_path_prob, log_action_probs, parent_id, parent_action) where {S} =
    BFSPathNode{S}(id, state, Float32(path_cost), Float32(log_path_prob), Float32.(log_action_probs), parent_id, parent_action)
BFSPathNode(id, state::S, path_cost, log_path_prob, log_action_probs) where {S} =
    BFSPathNode{S}(id, state, Float32(path_cost), Float32(log_path_prob), Float32.(log_action_probs), nothing, nothing)

function reconstruct(node_id::UInt, search_tree::Dict{UInt,BFSPathNode{S}}) where S
    plan, traj = Term[], S[]
    while node_id in keys(search_tree)
        node = search_tree[node_id]
        pushfirst!(traj, node.state)
        if node.parent_id === nothing break end
        pushfirst!(plan, node.parent_action)
        node_id = node.parent_id
    end
    return plan, traj
end



SymbolicPlanners.@auto_hash_equals mutable struct BFSPathSearchSolution{
    S <: State, T
} <: AbstractPathSearchSolution
    "Status of the returned solution."
    status::Symbol
    "Sequence of actions that reach the goal. May be partial / incomplete."
    plan::Vector{Term}
    "Trajectory of states that will be traversed while following the plan."
    trajectory::Union{Vector{S},Nothing}
    "Number of nodes expanded during search."
    expanded::Int
    "Tree of [`BFSPathNode`](@ref)s expanded or evaluated during search."
    search_tree::Union{Dict{UInt,BFSPathNode{S}},Nothing}
    "Frontier of yet-to-be-expanded search nodes (stored as references)."
    search_frontier::T
    "Order of nodes expanded during search (stored as references)."
    search_order::Vector{UInt}
end

BFSPathSearchSolution(status::Symbol, plan) =
    BFSPathSearchSolution(status, convert(Vector{Term}, plan), State[],
                       -1, nothing, nothing, UInt[])
BFSPathSearchSolution(status::Symbol, plan, trajectory) =
    BFSPathSearchSolution(status, convert(Vector{Term}, plan), trajectory,
                       -1, nothing, nothing, UInt[])

function Base.copy(sol::BFSPathSearchSolution)
    plan = copy(sol.plan)
    trajectory = isnothing(sol.trajectory) ?
        nothing : copy(sol.trajectory)
    search_tree = isnothing(sol.search_tree) ?
        nothing : copy(sol.search_tree)
    search_frontier = isnothing(sol.search_frontier) ?
        nothing : copy(sol.search_frontier)
    search_order = copy(sol.search_order)
    return BFSPathSearchSolution(sol.status, plan, trajectory, sol.expanded,
                              search_tree, search_frontier, search_order)
end

function Base.show(io::IO, m::MIME"text/plain", sol::BFSPathSearchSolution)
    # Invoke call to Base.show for AbstractBFSPathSearchSolution
    invoke(show, Tuple{IO, typeof(m), AbstractBFSPathSearchSolution}, io, m, sol)
    # Print search information if present
    if !isnothing(sol.search_tree)
        print(io, "\n  expanded: ", sol.expanded)
        print(io, "\n  search_tree: ", summary(sol.search_tree))
        print(io, "\n  search_frontier: ", summary(sol.search_frontier))
        if !isempty(sol.search_order)
            print(io, "\n  search_order: ", summary(sol.search_order))
        end
    end
end



"""
    BFSPlanner(;
        heuristic::Heuristic = GoalCountHeuristic(),
        search_noise::Union{Nothing,Float64} = nothing,
        g_mult::Float32 = 1.0f0,
        h_mult::Float32 = 1.0f0,
        max_nodes::Int = typemax(Int),
        max_time::Float64 = Inf,
        save_search::Bool = false,
        save_search_order::Bool = false
    )

Forward best-first search planner, which encompasses uniform-cost search, 
greedy search, and A* search. Each node ``n`` is expanded in order of increasing
priority ``f(n)``, defined as:

```math
f(n) = g_\\text{mult} \\cdot g(n) + h_\\text{mult} \\cdot h(n)
```

where ``g(n)`` is the path cost from the initial state to ``n``, and ``h(n)``
is the heuristic's goal distance estimate.

Returns a [`BFSPathSearchSolution`](@ref) if the goal is achieved, containing a 
plan that reaches the goal node, and `status` set to `:success`. If the node
or time budget runs out, the solution will instead contain a partial plan to
the last node selected for expansion, with `status` set to `:max_nodes` or 
`:max_time` accordingly.

If `save_search` is true, the returned solution will contain the search tree
and frontier so far. If `save_search` is true and the search space is exhausted
return a `NullSolution` with `status` set to `:failure`.

# Arguments
"Search heuristic that estimates cost of a state to the goal."
heuristic::Heuristic = GoalCountHeuristic()
"Maximum number of search nodes before termination."
max_nodes::Int = typemax(Int)
"Maximum time in seconds before planner times out."
max_time::Float64 = Inf
"Flag to save the search tree and frontier in the returned solution."
save_search::Bool = false
"Flag to save the node expansion order in the returned solution."
save_search_order::Bool = false
"Use heuristic value in the search"
use_heuristic::Bool = false
"Use heuristic value provided by NN"
use_learned_heuristic::Bool = true
"an estimate of the solution length"
max_h::Float32 = 1000f0

"""
mutable struct BFSPlanner <: Planner
    heuristic::Heuristic
    max_nodes::Int
    max_time::Float64
    save_search::Bool
    save_search_order::Bool
    use_heuristic::Bool
    use_learned_heuristic::Bool
    max_h::Float32
end

function BFSPlanner(;heuristic = GoalCountHeuristic(), max_nodes = typemax(Int), max_time = typemax(Float64), save_search = false, save_search_order = false, use_heuristic = false, use_learned_heuristic = true, max_h = 1000f0)
    BFSPlanner(heuristic, max_nodes, max_time, save_search, save_search_order, use_heuristic, use_learned_heuristic, max_h)    
end

BFSPlanner(heuristic::Heuristic; kwargs...) =
    BFSPlanner(;heuristic=heuristic, kwargs...)

function SymbolicPlanners.solve(planner::BFSPlanner,
               domain::Domain, state::State, spec::Specification)
    heuristic = planner.heuristic
    # Simplify goal specification
    spec = SymbolicPlanners.simplify_goal(spec, domain, state)
    # Precompute heuristic information
    SymbolicPlanners.precompute!(heuristic, domain, state, spec)
    # Initialize search tree and priority queue
    node_id = hash(state)
    est_cost, log_action_probs = compute(heuristic, domain, state, spec)
    search_tree = Dict(node_id => BFSPathNode(node_id, state, 0.0, 0.0, log_action_probs))
    priority = (est_cost, est_cost, 0)
    queue = PriorityQueue(node_id => priority)
    search_order = UInt[]
    sol = BFSPathSearchSolution(:in_progress, Term[], Vector{typeof(state)}(),
                             0, search_tree, queue, search_order)
    # Run the search
    sol = SymbolicPlanners.search!(sol, planner, domain, spec)
    # Return solution
    if planner.save_search
        return sol
    elseif sol.status == :failure
        return NullSolution(sol.status)
    else
        return BFSPathSearchSolution(sol.status, sol.plan, sol.trajectory)
    end
end

function SymbolicPlanners.search!(sol::BFSPathSearchSolution, planner::BFSPlanner,
                 domain::Domain, spec::Specification)
    start_time = time()
    sol.expanded = 0
    queue, search_tree = sol.search_frontier, sol.search_tree
    while length(queue) > 0
        # Get state with lowest estimated cost to goal
        node_id, _ = peek(queue)
        node = search_tree[node_id]
        # Check search termination criteria
        if is_goal(spec, domain, node.state)
            sol.status = :success # Goal reached
        elseif sol.expanded >= planner.max_nodes
            sol.status = :max_nodes # Node budget reached
        elseif time() - start_time >= planner.max_time
            sol.status = :max_time # Time budget reached
        end
        if sol.status == :in_progress
            # Dequeue current node
            dequeue!(queue)
            # Expand current node
            SymbolicPlanners.expand!(planner, node, search_tree, queue, domain, spec)
            sol.expanded += 1
            if planner.save_search && planner.save_search_order
                push!(sol.search_order, node_id)
            end
        else # Reconstruct plan and return solution
            sol.plan, sol.trajectory = reconstruct(node_id, search_tree)
            return sol
        end
    end
    sol.status = :failure
    return sol
end

function SymbolicPlanners.expand!(planner::BFSPlanner, node::BFSPathNode,
                 search_tree::Dict{UInt,<:BFSPathNode}, queue::PriorityQueue,
                 domain::Domain, spec::Specification)
    heuristic = planner.heuristic
    action2id = planner.heuristic.pddle.action2id

    state = node.state
    log_action_probs = node.log_action_probs
    # Iterate over available actions
    for act in available(domain, state)
        # Execute action and trigger all post-action events
        next_state = execute(domain, state, act; check=false)
        next_id = hash(next_state)
        # Check if next state satisfies trajectory constraints
        if is_violated(spec, domain, state) continue end
        # Compute path cost
        act_cost = get_cost(spec, domain, state, act, next_state)
        path_cost = node.path_cost + act_cost
        log_path_prob = node.log_path_prob + log_action_probs[get_action_id(action2id, domain, act)]
        predicted_h, next_action_probs = compute(planner.heuristic, domain, next_state, spec)
        # Update path costs if new path is shorter
        next_node = get!(search_tree, next_id,
                         BFSPathNode(next_id, next_state, Inf32, log_path_prob, next_action_probs))
        cost_diff = next_node.path_cost - path_cost
        if cost_diff > 0
            next_node.parent_id = node.id
            next_node.parent_action = act
            next_node.path_cost = path_cost
            # Update estimated cost from next state to goal
            if !(next_id in keys(queue))
                levin_cost = get_levin_cost(path_cost, log_path_prob, predicted_h)
                priority = (levin_cost, predicted_h, length(search_tree))
                enqueue!(queue, next_id, priority)
            else
                f_val, predicted_h, n_nodes = queue[next_id]
                queue[next_id] = (f_val - cost_diff, predicted_h, n_nodes)
            end
        end
    end
end

function get_action_id(action2id, domain,  a)
    actions = domain.actions
    assignment = Dict(zip(actions[a.name].args, a.args))
    predicates = NeuroPlanner.extract_predicates(actions[a.name])
    k = (a.name, [NeuroPlanner.ground(p, assignment) for p in predicates]...)
    action2id[k]
end

function get_levin_cost(path_cost, log_path_prob, predicted_h)
    predicted_h = max(0, predicted_h)
    log(predicted_h + path_cost) - log_path_prob
end
