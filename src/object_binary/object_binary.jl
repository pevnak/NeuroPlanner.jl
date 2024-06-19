"""
struct ObjectBinary{DO,P,EB,MP,D,II,S,G}
    domain::DO
    predicates::P
    edgebuilder::EB
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol,Int64}
    model_params::MP
    obj2id::D
    cached_types::II
    init_state::S
    goal_state::G
end


Represents STRIPS state as a graph, where vertices corresponds to objects and contants 
constants (map is stored in `obj2id`). There is an edge between objects, if they are both
in the same atom with arity higher than two. The type of the atom (predicate) is considered
as a type / feature of the edge. Unary and nullary predicates are represented as feature 
of the vertex / all vertices.

Following parameters are initialized for domain when `ObjectAtomBip(domain)` is called

* `predicates` contains information about predicates as produced by `PredicateInfo(domain)`. 
    It contains list of predicates, their arrities, number of predicates with arity higher than two,
    and number of predicates with arity 0 and 1. Finally, it contains a map of predicate names to 
    indices, which is used in `intstates` to convert names to ids
* `objtype2id` maps object types to an index in one-hot encoded vertex' properties 
* `constmap` maps constants to an index in one-hot encoded vertex' properties 
* `model_params` some parameters of an algorithm constructing the message passing passes 
* `edgebuilder` is used to specify, if different types of edges are represented as edges 
with features (use `FeaturedEdgeBuilder`) or use multigraph with edges of multiple types (use `MultiEdgeBuilder`).

Following parameters are initiliazed when extractor is specialized per problem instance

* `obj2id` contains a map from object names to their ids. This is created when calling `specialize(ex::ObjectAtom, problem)`
* `init_state` is information about initial state. If `nothing` it is added to the represenation of the state allowing a 
    to measure heuristic distance between initial_state and given state
* `init_state` is information about goal state. If `nothing` it is added to the represenation of the state allowing a 
    to measure heuristic distance between goal_state and given state
* `cached_types` contains a cache of (objectid, types), which makes the repeated extraction of states from the same
problem instance faster.

Constructors:

* `ObjectBinary(domain; message_passes=2, residual=:linear, edgebuilder = FeaturedEdgeBuilder, kwargs...)` The basic constructor 
specializing for a given domain. Notice the arguments `message_passes` and 
`residual` specifying the number of layers and the use of residual connections. 
Furthermore, there is an `edgebuilder` specifying how edges will be represented.
* `ObjectBinary(domain, problem; embed_goal=true, kwargs...)` specializes the constructor for a 
given domain and problem instance. The arguments are `message_passes`, `residual`, and additional `embed_goal` which will
automatically extract information about the goal state.

We define three convenient shortcuts to different variants of representation of edges: 

* `ObjectBinaryFE` represents different types of edges as edges with features. The edges 
are deduplicated, which means that if there are more edges, between two vertices, then they
will be represented as a single edge with features aggregated (by default by `max`).
* `ObjectBinaryFENA` is similar as ``ObjectBinaryFE`, but the edges are not deduplicated.
* `ObjectBinaryME` represents different edges as multi-graph.
"""
struct ObjectBinary{DO,P,EB,MP,D,II,S,G}
    domain::DO
    predicates::P
    edgebuilder::EB
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol,Int64}
    model_params::MP
    obj2id::D
    cached_types::II
    init_state::S
    goal_state::G
    function ObjectBinary(domain::DO, predicates::P, edgebuilder::EB, objtype2id::Dict{Symbol,Int64}, 
        constmap::Dict{Symbol,Int64}, model_params::MP, obj2id::D, cached_types::II, init::S, goal::G) where {DO,P,EB,MP<:NamedTuple,D,II,S,G}
        @assert issubset((:message_passes, :residual), keys(model_params)) "Parameters of the model are not fully specified"
        @assert (init === nothing || goal === nothing) "Fixing init and goal state is bizzaare, as the extractor would always create a constant"
        new{DO,P,EB,MP,D,II,S,G}(domain, predicates, edgebuilder, objtype2id, constmap, model_params, obj2id, cached_types, init, goal)
    end
end


ObjectBinaryNoGoal{DO,P,EB,MP,II} = ObjectBinary{DO,P,EB,MP,D,II,Nothing,Nothing} where {DO,P,EB,MP,D,II}
ObjectBinaryStart{DO,P,EB,MP,D,II,S} = ObjectBinary{DO,P,EB,MP,D,II,S,Nothing} where {DO,P,EB,MP,D,II,S<:Vector}
ObjectBinaryGoal{DO,P,EB,MP,D,II,G} = ObjectBinary{DO,P,EB,MP,D,II,Nothing,G} where {DO,P,EB,MP,D,II,G<:Vector}

function ObjectBinary(domain; message_passes=2, residual=:linear, edgebuilder = FeaturedEdgeBuilder, kwargs...)
    any(length(p.args) > 3 for p in values(domain.predicates)) && error("Ternary predicate is rare and it is not supported at the moment")
    model_params = (; message_passes, residual)
    dictmap(x) = Dict(reverse.(enumerate(sort(x))))
    pifo = PredicateInfo(domain)
    
    # now we want to set ids of atoms such that nunary predicates have own ids and nary predicates have their own.
    m01 = pifo.arrities .< 2
    m2 = pifo.arrities .≥ 2
    pifo = @set pifo.id2fid = cumsum(m01) .* m01 .+ cumsum(m2) .* m2

    # we might actually want to extract information about types one and forget about it
    objtype2id = dictmap(collect(keys(domain.typetree)))

    constmap = Dict{Symbol,Int}(dictmap([x.name for x in domain.constants]))
    ObjectBinary(domain, pifo, edgebuilder, objtype2id, constmap, model_params, nothing, nothing, nothing, nothing)
end

isspecialized(ex::ObjectBinary) = ex.obj2id !== nothing
hasgoal(ex::ObjectBinary) = ex.init_state !== nothing || ex.goal_state !== nothing

function ObjectBinary(domain, problem; embed_goal=true, kwargs...)
    ex = ObjectBinary(domain; kwargs...)
    ex = specialize(ex, problem)
    embed_goal ? add_goalstate(ex, problem) : ex
end

ObjectBinaryFE(domain; kwargs...) = ObjectBinary(domain; edgebuilder = FeaturedHyperEdgeBuilder, kwargs...)
ObjectBinaryFENA(domain; kwargs...) = ObjectBinary(domain; edgebuilder = FeaturedHyperEdgeBuilderNA, kwargs...)
ObjectBinaryME(domain; kwargs...) = ObjectBinary(domain; edgebuilder = MultiHyperEdgeBuilder, kwargs...)
ObjectBinaryFE(domain, problem; kwargs...) = ObjectBinary(domain, problem; edgebuilder = FeaturedHyperEdgeBuilder, kwargs...)
ObjectBinaryFENA(domain, problem; kwargs...) = ObjectBinary(domain, problem; edgebuilder = FeaturedHyperEdgeBuilderNA, kwargs...)
ObjectBinaryME(domain, problem; kwargs...) = ObjectBinary(domain, problem; edgebuilder = MultiHyperEdgeBuilder, kwargs...)

function Base.show(io::IO, ex::ObjectBinary)
    if !isspecialized(ex)
        print(io, "Unspecialized extractor for ", ex.domain.name, " ",ex.predicates.predicates)
    else
        g = hasgoal(ex) ? "with" : "without"
        print(io, "Specialized extractor ", g, " goal for ", ex.domain.name, " ", ex.predicates.predicates)
    end
end


"""
specialize(ex::ObjectBinary{<:Nothing,<:Nothing}, problem)

initializes extractor for a given `problem` by initializing mapping 
from objects to id of vertices. Goals are not changed added to the 
extractor.
"""
function specialize(ex::ObjectBinary, problem)
    obj2id = Dict(v.name => i for (i, v) in enumerate(problem.objects))
    for k in keys(ex.constmap)
        obj2id[k] = length(obj2id) + 1
    end
    cached_types = cache_types_const(ex, obj2id, problem)
    ex = ObjectBinary(ex.domain, ex.predicates, ex.edgebuilder, ex.objtype2id, ex.constmap, ex.model_params, obj2id, cached_types, nothing, nothing)
end


function intstates(ex::ObjectBinary, state::GenericState)
    facts = collect(PDDL.get_facts(state))
    intstates(ex.domain, ex.obj2id, ex.predicates, facts)
end

function (ex::ObjectBinary)(state::GenericState)
    # we need to add goal of start state with correctly remapped ids
    grouped_facts = addgoal(ex, intstates(ex, state))
    encode_state(ex, state, grouped_facts)
end

function encode_state(ex::ObjectBinary, state::GenericState, grouped_facts)
    message_passes = ex.model_params.message_passes
    residual = ex.model_params.residual
    x = nunary_predicates(ex, state, grouped_facts)
    kb = KnowledgeBase((; x1=x))
    n = size(x, 2)
    sₓ = :x1
    edge_structure = multi_predicates(ex, :x1, grouped_facts)
    if ex.predicates.nary > 0
        for i in 1:message_passes
            input_to_gnn = last(keys(kb))
            ds = KBEntryRenamer(:x1, input_to_gnn)(edge_structure)
            kb = append(kb, layer_name(kb, "gnn"), ds)
            if residual !== :none #if there is a residual connection, add it 
                kb = add_residual_layer(kb, keys(kb)[end-1:end], n)
            end
        end
    end
    s = last(keys(kb))
    kb = append(kb, :o, BagNode(ArrayNode(KBEntry(s, 1:n)), [1:n]))
end


"""
    cache_types_const(ex::ObjectBinary)

    Compute indices corresponding to types of objects and constants, which 
    are fixed for the problem. Indices are stores as a Vector of tuples 
    containing row and col to be set
"""
function cache_types_const(ex::ObjectBinary, obj2id, problem)
    ii = NTuple{2,Int}[]
    # encode types of objects
    offset = ex.predicates.nunary
    for (s, t) in problem.objtypes
        i = ex.objtype2id[t]
        j = obj2id[s.name]
        push!(ii, (offset + i, j))
    end

    # encode constants
    offset = ex.predicates.nunary + length(ex.objtype2id)
    for (k, i) in ex.constmap
        j = obj2id[k]
        push!(ii, (offset + i, j))
    end
    ii
end


"""
nunary_predicates(ex::ObjectBinary, state)

Create matrix with one column per object and encode by one-hot-encoding unary predicates 
and types of objects. Nunary predicates are encoded as properties of all objects.
"""
function nunary_predicates(ex::ObjectBinary, state, grouped_facts)
    idim = ex.predicates.nunary + length(ex.objtype2id) + length(ex.constmap)
    x = zeros(Float32, idim, length(ex.obj2id))

    # Set features indicating type of the object and constant.
    # The indices were precalculated during initialization.
    for (row, col) in ex.cached_types
        x[row,col] = 1
    end

    # unary predicates
    for s in grouped_facts[2] 
        x[s.name, s.args[1]] = 1
    end

    # nullary predicates
    for s in grouped_facts[1] 
        x[s.name, :] .= 1
    end
    x
end


function multi_predicates(ex::ObjectBinary, kid::Symbol, grouped_facts, prefix=nothing)
    # estimate the number of predicates and initiates the 
    max_edges = mapreduce(+, 3:length(grouped_facts)) do i
        a = i - 1 # arity of the predicate
        (a * (a - 1) ÷ 2 ) * length(grouped_facts[i])
    end
    num_type_edges = ex.predicates.nary
    num_vertices = length(ex.obj2id)
    eb = ex.edgebuilder(Val(2), max_edges, num_vertices, num_type_edges)

    # predicates from edges
    for preds in grouped_facts[3:end]
        encode_predicates!(eb, ex, preds)
    end
    return(construct(eb, kid))
end

"""
encode_predicates(ex::ObjectBinary, pname::Symbol, preds, kid::Symbol)

Encodes predicates for an ObjectBinary instance.

Arguments:
- ex::ObjectBinary: ObjectBinary instance
- pname::Symbol: Predicate name
- preds: Predicates
- kid::Symbol: Symbol representing the key ID

Returns:
- BagNode: Encoded predicates as a BagNode

This function encodes predicates for an ObjectBinary instance using the given predicate name, predicates, and key ID.
"""
function encode_predicates!(eb::EB, ex::ObjectBinary, preds::Vector{<:IntState{N,<:Integer}}) where {N, EB<:Union{FeaturedEdgeBuilder,MultiEdgeBuilder}}
    for p in preds
        for i in 1:N
            oᵢ = p.args[i]
            for j in i+1:N
                oⱼ = p.args[j]
                @inbounds push!(eb, (oᵢ, oⱼ), p.name)
            end
        end
    end
end


"""
    goal_id2fid(ex::ObjectBinary)

    construct new id2fid, such feature-ids of predicates points behind
    the feature_ids of normal states. 
"""
function goal_id2fid(ex::ObjectBinary)
    arrities = ex.predicates.arrities
    m01 = arrities .≤ 1
    m2 = arrities .> 1
    m01 .* sum(m01) .+ m2 .* sum(m2)
    id2fid = ex.predicates.id2fid .+ m01 .* sum(m01) .+ m2 .* sum(m2)
end

function add_initstate(ex::ObjectBinary, problem, start=initstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)

    #extract the init state
    new_ex = @set ex.init_state = intstates(ex, start)

    #change id2fid to point behind
    new_ex = @set new_ex.predicates.id2fid = goal_id2fid(new_ex)

    # change the number of predicates, which has doubled
    new_ex = @set new_ex.predicates.nunary = 2*new_ex.predicates.nunary
    new_ex = @set new_ex.predicates.nary = 2*new_ex.predicates.nary
    new_ex
end

function add_goalstate(ex::ObjectBinary, problem, goal=goalstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)

    # change id2fid to code the goal, extract goal state with it and set it to goal_states
    gex = @set ex.predicates.id2fid = goal_id2fid(ex)
    new_ex = @set ex.goal_state = intstates(gex, goal)

    # change the number of predicates, which has doubled
    new_ex = @set new_ex.predicates.nunary = 2*new_ex.predicates.nunary
    new_ex = @set new_ex.predicates.nary = 2*new_ex.predicates.nary
    new_ex
end

function addgoal(ex::ObjectBinaryStart, kb)
    return (merge_states(ex.init_state, kb))
end

function addgoal(ex::ObjectBinaryGoal, kb)
    return (merge_states(kb, ex.goal_state))
end

function addgoal(ex::ObjectBinaryNoGoal, kb)
    return (kb)
end
