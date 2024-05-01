using PDDL: get_facts, get_args
"""
struct ObjectBinary{DO,D,N,MP,S,G}
    domain::DO
    multiarg_predicates::NTuple{N,Symbol}
    nunanary_predicates::Dict{Symbol,Int64}
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol,Int64}
    model_params::MP
    obj2id::D
    init_state::S
    goal_state::G
end

Represents a PDDL state as a multigraph, where 
- Each node is either an object or a contant
- unary predicate is a property of an object
- nullary predicate is a property of all objects
- n-ary predicate is an edge between two vertices / objects which are 

The computational model is message-passing over hyper-graph, which is essentially 
a message passing over a bipartite graph, where left vertices corresponds to vertices
in hypergraph and right vertices corresponds to hyper-edges. There is an edge between
vertex corresponding to the hyper-edge and its vertices.

--- `multiarg_predicates` is a list of all n-ary predicates
--- `nunanary_predicates` maps unary and nullary predicates to an index in one-hot encoded vertex' properties 
--- `objtype2id` maps object types to an index in one-hot encoded vertex' properties 
--- `constmap` maps constants to an index in one-hot encoded vertex' properties 
--- `model_params` some parameters of an algorithm constructing the message passing passes 
"""

struct ObjectBinary{DO,D,N,MP,S,G}
    domain::DO
    multiarg_predicates::NTuple{N,Symbol}
    nunanary_predicates::Dict{Symbol,Int64}
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol,Int64}
    model_params::MP
    obj2id::D
    init_state::S
    goal_state::G
    function ObjectBinary(domain::DO, multiarg_predicates::NTuple{N,Symbol}, nunanary_predicates::Dict{Symbol,Int64},
        objtype2id::Dict{Symbol,Int64}, constmap::Dict{Symbol,Int64}, model_params::MP, obj2id::D, init::S, goal::G) where {DO,D,N,MP<:NamedTuple,S,G}

        @assert issubset((:message_passes, :residual), keys(model_params)) "Parameters of the model are not fully specified"
        @assert (init === nothing || goal === nothing) "Fixing init and goal state is bizzaare, as the extractor would always create a constant"
        new{DO,D,N,MP,S,G}(domain, multiarg_predicates, nunanary_predicates, objtype2id, constmap, model_params, obj2id, init, goal)
    end
end


ObjectBinaryNoGoal{DO,D,N,MP} = ObjectBinary{DO,D,N,MP,Nothing,Nothing} where {DO,D,N,MP}
ObjectBinaryStart{DO,D,N,MP,S} = ObjectBinary{DO,D,N,MP,S,Nothing} where {DO,D,N,MP,S<:KnowledgeBase}
ObjectBinaryGoal{DO,D,N,MP,S} = ObjectBinary{DO,D,N,MP,Nothing,S} where {DO,D,N,MP,S<:KnowledgeBase}

function ObjectBinary(domain; message_passes=2, residual=:linear, kwargs...)
    model_params = (; message_passes, residual)
    dictmap(x) = Dict(reverse.(enumerate(sort(x))))
    predicates = collect(domain.predicates)
    multiarg_predicates = tuple([kv[1] for kv in predicates if length(kv[2].args) > 1]...)
    nunanary_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) ≤ 1])
    objtype2id = Dict(s => i + length(nunanary_predicates) for (i, s) in enumerate(collect(keys(domain.typetree))))
    constmap = Dict{Symbol,Int}(dictmap([x.name for x in domain.constants]))
    ObjectBinary(domain, multiarg_predicates, nunanary_predicates, objtype2id, constmap, model_params, nothing, nothing, nothing)
end

isspecialized(ex::ObjectBinary) = ex.obj2id !== nothing
hasgoal(ex::ObjectBinary) = ex.init_state !== nothing || ex.goal_state !== nothing

function ObjectBinary(domain, problem; embed_goal=true, kwargs...)
    ex = ObjectBinary(domain; kwargs...)
    ex = specialize(ex, problem)
    embed_goal ? add_goalstate(ex, problem) : ex
end

function Base.show(io::IO, ex::ObjectBinary)
    if !isspecialized(ex)
        print(io, "Unspecialized extractor for ", ex.domain.name, " (", length(ex.nunanary_predicates), ", ", length(ex.multiarg_predicates), ")")
    else
        g = hasgoal(ex) ? "with" : "without"
        print(io, "Specialized extractor ", g, " goal for ", ex.domain.name, " (", length(ex.nunanary_predicates), ", ", length(ex.multiarg_predicates), ", ", length(ex.obj2id), ")")
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
    ObjectBinary(ex.domain, ex.multiarg_predicates, ex.nunanary_predicates, ex.objtype2id, ex.constmap, ex.model_params, obj2id, nothing, nothing)
end

function (ex::ObjectBinary)(state::GenericState)
    prefix = (ex.goal_state !== nothing) ? :start : ((ex.init_state !== nothing) ? :goal : nothing)
    kb = encode_state(ex, state, prefix)
    addgoal(ex, kb)
end

function encode_state(ex::ObjectBinary, state::GenericState, prefix=nothing)
    message_passes, residual = ex.model_params
    grouped_facts = group_facts(ex, collect(PDDL.get_facts(state)))
    x = nunary_predicates(ex, state)
    kb = KnowledgeBase((; x1=x))
    n = size(x, 2)
    sₓ = :x1
    edge_structure = multi_predicates(ex, :x1, grouped_facts, prefix)
    if !isempty(ex.multiarg_predicates)
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
function nunary_predicates(ex::ObjectBinary, state)
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
        j = ex.obj2id[only(s.args).name]
        x[i, j] = 1
    end

    for f in filter(f -> length(get_args(f)) < 2, get_facts(state))
        v = 1
        if (f isa PDDL.Compound) && (f.name == :not)
            f = only(f.args)
            v = 0
        end
        a = get_args(f)
        pid = ex.nunanary_predicates[f.name]
        if length(a) == 1
            vid = ex.obj2id[only(get_args(f)).name]
            x[pid, vid] = v
        else
            length(a) == 0
            x[pid, :] .= v
        end
    end
    x
end


function multi_predicates(ex::ObjectBinary, kid::Symbol, grouped_facts, prefix=nothing)
    # Then, we specify the predicates the dirty way
    ks = ex.multiarg_predicates
    xs = map(kii -> encode_predicates(ex, kii[2], kid), grouped_facts)
    ns = isnothing(prefix) ? ks : _map_tuple(k -> Symbol(prefix, "_", k), ks)
    ProductNode(NamedTuple{ns}(xs))
end


function group_facts(ex::ObjectBinary, facts::Vector{<:Term})
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

function factargs2id(ex::ObjectBinary, facts, mask, arity::Val{N}) where {N}
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

function encode_predicates(ex::ObjectBinary, preds::Vector{NTuple{N,Int64}}, kid::Symbol) where {N}
    pred_length = Int(((N - 1) + 1) * (N - 1) / 2)
    eb = EdgeBuilderComp(2, pred_length * length(preds), length(ex.obj2id))
    for p in preds
        for i in 1:length(p)-1
            oᵢ = p[i]
            for j in i+1:length(p)
                oⱼ = p[j]
                @inbounds push!(eb, (oᵢ, oⱼ))
            end
        end
    end
    construct(eb, kid)
end

function add_goalstate(ex::ObjectBinary, problem, goal=goalstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    @set ex.goal_state = encode_state(ex, goal, :goal)
end

function add_initstate(ex::ObjectBinary, problem, start=initstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    @set ex.init_state = encode_state(ex, start, :start)
end

function addgoal(ex::ObjectBinaryStart, kb::KnowledgeBase)
    return (stack_hypergraphs(ex.init_state, kb))
end

function addgoal(ex::ObjectBinaryGoal, kb::KnowledgeBase)
    return (stack_hypergraphs(kb, ex.goal_state))
end

function addgoal(ex::ObjectBinaryNoGoal, kb::KnowledgeBase)
    return (kb)
end
