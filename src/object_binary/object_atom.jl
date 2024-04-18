using PDDL: get_facts, get_args
"""
struct ObjectAtom{DO,D,N,G}
    domain::DO
    multiarg_predicates::NTuple{N,Symbol}
    nunanary_predicates::Dict{Symbol,Int64}
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol, Int64}
    model_params::NamedTuple{(:message_passes, :residual), Tuple{Int64, Symbol}}
    obj2id::D
    goal::G
end

Represents a PDDL state as a hypergraph, whre 
- Each node is either an object or a contant
- unary predicate is a property of an object
- nullary predicate is a property of all objects
- n-ary predicate is a hyper-edge

The computational model is message-passing over hyper-graph, which is essentially 
a message passing over a bipartite graph, where left vertices corresponds to vertices
in hypergraph and right vertices corresponds to hyper-edges. There is an edge between
vertex corresponding to the hyper-edge and its vertices.

--- `multiarg_predicates` is a list of all n-ary predicates
--- `nunary_predicates` maps unary predicates to an index in one-hot encoded vertex' properties 
--- `objtype2id` maps unary predicates to an index in one-hot encoded vertex' properties 
--- `constmap` maps constants to an index in one-hot encoded vertex' properties 
--- `model_params` some parameters of an algorithm constructing the message passing passes 
"""
struct ObjectAtom{DO,D,N,M,MP,S,G}
    domain::DO
    multiarg_predicates::NTuple{N,Symbol}
    nunanary_predicates::NTuple{M,Symbol}
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol,Int64}
    model_params::MP
    obj2id::D
    init_state::S
    goal_state::G
    function ObjectAtom(domain::DO, multiarg_predicates::NTuple{N,Symbol}, nunanary_predicates::NTuple{M,Symbol}, objtype2id::Dict{Symbol,Int64}, constmap::Dict{Symbol,Int64}, model_params::MP, obj2id::D, init::S, goal::G) where {DO,D,N,M,MP<:NamedTuple,S,G}
        @assert issubset((:message_passes, :residual), keys(model_params)) "Parameters of the model are not fully specified"
        @assert (init === nothing || goal === nothing) "Fixing init and goal state is bizzaare, as the extractor would always create a constant"
        new{DO,D,N,M,MP,S,G}(domain, multiarg_predicates, nunanary_predicates, objtype2id, constmap, model_params, obj2id, init, goal)
    end
end

ObjectAtomNoGoal{DO,D,N,M,MP} = ObjectAtom{DO,D,N,M,MP,Nothing,Nothing} where {DO,D,N,M,MP}
ObjectAtomStart{DO,D,N,M,MP,S} = ObjectAtom{DO,D,N,M,MP,S,Nothing} where {DO,D,N,M,MP,S<:KnowledgeBase}
ObjectAtomGoal{DO,D,N,M,MP,S} = ObjectAtom{DO,D,N,M,MP,Nothing,S} where {DO,D,N,M,MP,S<:KnowledgeBase}

function ObjectAtom(domain; message_passes=2, residual=:linear, kwargs...)
    model_params = (; message_passes, residual)
    dictmap(x) = Dict(reverse.(enumerate(sort(x))))
    predicates = collect(domain.predicates)
    multiarg_predicates = tuple([kv[1] for kv in predicates if length(kv[2].args) > 1]...)
    nunanary_predicates = tuple(sort([kv[1] for kv in predicates if length(kv[2].args) ≤ 1])...)
    objtype2id = Dict(s => i + length(nunanary_predicates) for (i, s) in enumerate(collect(keys(domain.typetree))))
    constmap = Dict{Symbol,Int}(dictmap([x.name for x in domain.constants]))
    ObjectAtom(domain, multiarg_predicates, nunanary_predicates, objtype2id, constmap, model_params, nothing, nothing, nothing)
end

isspecialized(ex::ObjectAtom) = ex.obj2id !== nothing
hasgoal(ex::ObjectAtom) = ex.init_state !== nothing || ex.goal_state !== nothing


function ObjectAtom(domain, problem; embed_goal=true, kwargs...)
    ex = ObjectAtom(domain; kwargs...)
    ex = specialize(ex, problem)
    embed_goal ? add_goalstate(ex, problem) : ex
end

function Base.show(io::IO, ex::ObjectAtom)
    if !isspecialized(ex)
        print(io, "Unspecialized ObjectAtom extractor for ", ex.domain.name, " (", length(ex.nunanary_predicates), ", ", length(ex.multiarg_predicates), ")")
    else
        g = hasgoal(ex) ? "with" : "without"
        print(io, "Specialized ObjectAtom extractor ", g, " goal for ", ex.domain.name, " (", length(ex.nunanary_predicates), ", ", length(ex.multiarg_predicates), ", ", length(ex.obj2id), ")")
    end
end

"""
specialize(ex::ObjectAtom{<:Nothing,<:Nothing}, problem)

initializes extractor for a given `problem` by initializing mapping 
from objects to id of vertices. Goals are not changed added to the 
extractor.
"""
function specialize(ex::ObjectAtom, problem)
    obj2id = Dict(v.name => i for (i, v) in enumerate(problem.objects))
    for k in keys(ex.constmap)
        obj2id[k] = length(obj2id) + 1
    end
    ObjectAtom(ex.domain, ex.multiarg_predicates, ex.nunanary_predicates, ex.objtype2id, ex.constmap, ex.model_params, obj2id, nothing, nothing)
end

function (ex::ObjectAtom)(state::GenericState)
    prefix = (ex.goal_state !== nothing) ? :start : ((ex.init_state !== nothing) ? :goal : nothing)

    gf = group_facts(ex, collect(PDDL.get_facts(state)))
    kb = encode_state(ex, state, gf, prefix)
    addgoal(ex, kb)
end

function encode_state(ex::ObjectAtom, state, gf, prefix=nothing)
    message_passes, residual = ex.model_params
    x = nunary_predicates(ex, state, gf)
    kb = KnowledgeBase((; x1=x))
    n = size(x, 2)
    sₓ = :x1
    edge_structure = multi_predicates(ex, :x1, gf, prefix)
    kb = append(kb, layer_name(kb, "gnn"), edge_structure)
    ds = KBEntryRenamer(:x1, :gnn_2)(edge_structure)
    kb = append(kb, layer_name(kb, "gnn"), edge_structure)
    # if !isempty(ex.multiarg_predicates)
    #     for i in 1:message_passes
    #         input_to_gnn = last(keys(kb))
    #         ds = KBEntryRenamer(:x1, input_to_gnn)(edge_structure)
    #         kb = append(kb, layer_name(kb, "gnn"), ds)
    #         if residual !== :none #if there is a residual connection, add it 
    #             kb = add_residual_layer(kb, keys(kb)[end-1:end], n)
    #         end
    #     end
    # end
    s = last(keys(kb))
    kb = append(kb, :o, BagNode(ArrayNode(KBEntry(s, 1:n)), [1:n]))
end

"""
nunary_predicates(ex::ObjectAtom, state)

Create matrix with one column per object and encode by one-hot-encoding unary predicates 
and types of objects. Nunary predicates are encoded as properties of all objects.
"""
function nunary_predicates(ex::ObjectAtom, state, nunar_facts)
    # first, we completely specify the matrix with properties
    idim = length(ex.nunanary_predicates) + length(ex.objtype2id) + length(ex.constmap)
    x = zeros(Float32, idim, length(ex.obj2id))

    # encode constants
    offset = length(ex.nunanary_predicates) + length(ex.objtype2id)
    for (k, i) in ex.constmap
        j = ex.obj2id[k]
        x[offset+i, j] = 1
    end

    # encode types of objects
    for s in state.types
        i = ex.objtype2id[s.name]
        j = ex.obj2id[s.args[1].name]
        x[i, j] = 1
    end

    # encode unary predicates
    for (row, k) in enumerate(ex.nunanary_predicates)
        @inbounds for col in nunar_facts[k]
            x[row, col[1]] = 1
        end
    end
    x
end

function group_facts(ex::ObjectAtom, facts::Vector{<:Term})
    ns = tuple(ex.nunanary_predicates..., ex.multiarg_predicates...)
    occurences = falses(length(facts), length(ns))
    for (i, f) in enumerate(facts)
        col = _inlined_search(f.name, ns)
        occurences[i, col] = true
    end

    xs = _mapenumerate_tuple(ns) do col, k
        N = length(ex.domain.predicates[k].args)
        if N > 0 
            factargs2id(ex, facts, (@view occurences[:, col]), Val(N))
        else 
            (any(@view occurences[:,col]) ? (1:length(ex.obj2id)) : (0:-1))
        end
    end
    NamedTuple{ns}(xs)
end

function group_multi_facts(ex::ObjectAtom, facts::Vector{<:Term})
    occurences = falses(length(facts), length(ex.multiarg_predicates))
    for (i, f) in enumerate(facts)
        col = _inlined_search(f.name, ex.multiarg_predicates)
        col == -1 && continue
        occurences[i, col] = true
    end

    _mapenumerate_tuple(ex.multiarg_predicates) do col, k
        N = length(ex.domain.predicates[k].args)
        k => factargs2id(ex, facts, (@view occurences[:, col]), Val(N))
    end
end

function group_nunary_facts(ex::ObjectAtom, facts::Vector{<:Term})
    occurences = falses(length(facts), length(ex.nunanary_predicates))
    for (i, f) in enumerate(facts)
        col = _inlined_search(f.name, ex.nunanary_predicates)
        col == -1 && continue
        occurences[i, col] = true
    end

    xs = _mapenumerate_tuple(ex.nunanary_predicates) do col, k
        if length(ex.domain.predicates[k].args) == 1
            return(factargs2id(ex, facts, (@view occurences[:, col]), Val(1)))
        else
            (any(@view occurences[:,col]) ? (1:length(ex.obj2id)) : (0:-1))
        end
    end
    NamedTuple{ex.nunanary_predicates}(xs)
end

function factargs2id(ex::ObjectAtom, facts, mask, arity::Val{N}) where {N}
    d = ex.obj2id
    o = Vector{NTuple{N,Int}}(undef, sum(mask))
    index = 1
    for i in 1:length(mask)
        mask[i] || continue
        p = facts[i]
        o[index] = _map_tuple(j -> d[p.args[j].name], arity)
        index += 1
    end
    o
end
function multi_predicates(ex::ObjectAtom, kid::Symbol, grouped_facts, prefix=nothing)
    # Then, we specify the predicates the dirty way
    ks = ex.multiarg_predicates
    xs = map(k -> encode_predicates(ex, grouped_facts[k], kid), ks)
    ns = isnothing(prefix) ? ks : _map_tuple(k -> Symbol(prefix, "_", k), ks)
    ProductNode(NamedTuple{ns}(xs))
end

function encode_predicates(ex::ObjectAtom, preds::Vector{NTuple{N,Int64}}, kid::Symbol) where {N}
    eb = EdgeBuilderCompMat(Val(N), length(preds), length(ex.obj2id))
    for p in preds
        push!(eb, p)
    end
    construct(eb, kid)
end

function add_goalstate(ex::ObjectAtom, problem, goal=goalstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    @set ex.goal_state = encode_state(ex, goal, :goal)
end

function add_initstate(ex::ObjectAtom, problem, start=initstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    @set ex.init_state = encode_state(ex, start, :start)
end

function addgoal(ex::ObjectAtomStart, kb::KnowledgeBase)
    return (stack_hypergraphs(ex.init_state, kb))
end

function addgoal(ex::ObjectAtomGoal, kb::KnowledgeBase)
    return (stack_hypergraphs(kb, ex.goal_state))
end

function addgoal(ex::ObjectAtomNoGoal, kb::KnowledgeBase)
    return (kb)
end
