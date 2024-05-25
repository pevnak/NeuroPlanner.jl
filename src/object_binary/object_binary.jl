"""
struct ObjectBinary{DO,P,EB,MP,D,S,G}
    domain::DO
    predicates::P
    edgebuilder::EB
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol,Int64}
    model_params::MP
    obj2id::D
    init_state::S
    goal_state::G
end

Represents a PDDL state as a multigraph (or graph with features on edges), where 
- Each node is either an object or a contant
- unary predicate is a property of an object
- nullary predicate is a property of all objects
- n-ary predicate is an edge between two vertices (objects) sharing the object

The computational model is message-passing over hyper-graph, which is essentially 
a message passing over a bipartite graph, where left vertices corresponds to vertices
in hypergraph and right vertices corresponds to hyper-edges. There is an edge between
vertex corresponding to the hyper-edge and its vertices. 

--- `predicates` contains information about predicates as produced by `PredicateInfo(domain)`. 
    It contains list of predicates, their arrities, number of predicates with arity higher than two,
    and number of predicates with arity 0 and 1. Finally, it contains a map of predicate names to 
    indices, which is used in `intstates` to convert names to ids
--- `objtype2id` maps object types to an index in one-hot encoded vertex' properties 
--- `constmap` maps constants to an index in one-hot encoded vertex' properties 
--- `model_params` some parameters of an algorithm constructing the message passing passes 
--- `obj2id` maps object names to id of vertices
--- `init_state` fluents from the initial parsed to `intstates`. 
--- `goal_state` fluents from the initial parsed to `intstates` with ids appropriately shifted.

We define three variants: 
`ObjectBinaryFE` --- representing multiple edges as edges with features
`ObjectBinaryFENA` --- representing multiple edges as edges with features, but do not aggregate multiple edges
`ObjectBinaryME` --- representing edges as proper multi-graph
"""

struct ObjectBinary{DO,P,EB,MP,D,S,G}
    domain::DO
    predicates::P
    edgebuilder::EB
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol,Int64}
    model_params::MP
    obj2id::D
    init_state::S
    goal_state::G
    function ObjectBinary(domain::DO, predicates::P, edgebuilder::EB, objtype2id::Dict{Symbol,Int64}, 
        constmap::Dict{Symbol,Int64}, model_params::MP, obj2id::D, init::S, goal::G) where {DO,P,EB,MP<:NamedTuple,D,S,G}
        @assert issubset((:message_passes, :residual), keys(model_params)) "Parameters of the model are not fully specified"
        @assert (init === nothing || goal === nothing) "Fixing init and goal state is bizzaare, as the extractor would always create a constant"
        new{DO,P,EB,MP,D,S,G}(domain, predicates, edgebuilder, objtype2id, constmap, model_params, obj2id, init, goal)
    end
end


ObjectBinaryNoGoal{DO,P,edgebuilder,MP} = ObjectBinary{DO,P,edgebuilder,MP,D,Nothing,Nothing} where {DO,P,edgebuilder,MP,D}
ObjectBinaryStart{DO,P,edgebuilder,MP,D,S} = ObjectBinary{DO,P,edgebuilder,MP,S,D,Nothing} where {DO,P,edgebuilder,MP,D,S<:Vector}
ObjectBinaryGoal{DO,P,edgebuilder,MP,D,G} = ObjectBinary{DO,P,edgebuilder,MP,D,Nothing,G} where {DO,P,edgebuilder,MP,D,G<:Vector}

predicates_of_length(domain::GenericDomain, l) = tuple(filter(k -> length(domain.predicates[k].args) == l, keys(domain.predicates))...)

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
    ObjectBinary(domain, pifo, edgebuilder, objtype2id, constmap, model_params, nothing, nothing, nothing)
end

ObjectBinaryFE(domain;kwargs...) = ObjectBinary(domain;edgebuilder = FeaturedEdgeBuilder, kwargs...)
ObjectBinaryFENA(domain;kwargs...) = ObjectBinary(domain;edgebuilder = FeaturedEdgeBuilderNA, kwargs...)
ObjectBinaryME(domain;kwargs...) = ObjectBinary(domain;edgebuilder = MultiEdgeBuilder, kwargs...)

isspecialized(ex::ObjectBinary) = ex.obj2id !== nothing
hasgoal(ex::ObjectBinary) = ex.init_state !== nothing || ex.goal_state !== nothing

function ObjectBinary(domain, problem; embed_goal=true, kwargs...)
    ex = ObjectBinary(domain; kwargs...)
    ex = specialize(ex, problem)
    embed_goal ? add_goalstate(ex, problem) : ex
end

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
    ex = ObjectBinary(ex.domain, ex.predicates, ex.edgebuilder, ex.objtype2id, ex.constmap, ex.model_params, obj2id, nothing, nothing)
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
nunary_predicates(ex::ObjectBinary, state)

Create matrix with one column per object and encode by one-hot-encoding unary predicates 
and types of objects. Nunary predicates are encoded as properties of all objects.
"""
function nunary_predicates(ex::ObjectBinary, state, grouped_facts)
    idim = ex.predicates.nunary + length(ex.objtype2id) + length(ex.constmap)
    x = zeros(Float32, idim, length(ex.obj2id))

    # encode types of objects
    offset = ex.predicates.nunary
    for s in state.types
        i = ex.objtype2id[s.name]
        j = ex.obj2id[only(s.args).name]
        x[offset + i, j] = 1
    end

    # encode constants
    offset = ex.predicates.nunary + length(ex.objtype2id)
    for (k, i) in ex.constmap
        j = ex.obj2id[k]
        x[offset + i, j] = 1
    end

    # unary predicates
    for s in grouped_facts[2] 
        x[s.name, first(s.args)] = 1
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
    eb = ex.edgebuilder(2, max_edges, num_vertices, num_type_edges)

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
    new_ex = @set ex.predicates.id2fid = goal_id2fid(new_ex)

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
