using PDDL: get_facts, get_args
"""
struct MixedLRNN2{DO,D,N,G}
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
struct MixedLRNN2{DO,D,N,MP,S,G}
    domain::DO
    multiarg_predicates::NTuple{N,Symbol}
    nunanary_predicates::Dict{Symbol,Int64}
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol,Int64}
    model_params::MP
    obj2id::D
    init_state::S
    goal_state::G
    function MixedLRNN2(domain::DO, multiarg_predicates::NTuple{N,Symbol}, nunanary_predicates::Dict{Symbol,Int64}, objtype2id::Dict{Symbol,Int64}, constmap::Dict{Symbol,Int64}, model_params::MP, obj2id::D, init::S, goal::G) where {DO,D,N,MP<:NamedTuple,S,G}
        @assert issubset((:message_passes, :residual), keys(model_params)) "Parameters of the model are not fully specified"
        @assert (init === nothing || goal === nothing) "Fixing init and goal state is bizzaare, as the extractor would always create a constant"
        new{DO,D,N,MP,S,G}(domain, multiarg_predicates, nunanary_predicates, objtype2id, constmap, model_params, obj2id, init, goal)
    end
end

MixedLRNN2NoGoal{DO,D,N,MP} = MixedLRNN2{DO,D,N,MP,Nothing,Nothing} where {DO,D,N,MP}
MixedLRNN2Start{DO,D,N,MP,S} = MixedLRNN2{DO,D,N,MP,S,Nothing} where {DO,D,N,MP,S<:KnowledgeBase}
MixedLRNN2Goal{DO,D,N,MP,S} = MixedLRNN2{DO,D,N,MP,Nothing,S} where {DO,D,N,MP,S<:KnowledgeBase}

function MixedLRNN2(domain; message_passes=2, residual=:linear, kwargs...)
    model_params = (; message_passes, residual)
    dictmap(x) = Dict(reverse.(enumerate(sort(x))))
    predicates = collect(domain.predicates)
    multiarg_predicates = tuple([kv[1] for kv in predicates if length(kv[2].args) > 1]...)
    nunanary_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) ≤ 1])
    objtype2id = Dict(s => i + length(nunanary_predicates) for (i, s) in enumerate(collect(keys(domain.typetree))))
    constmap = Dict{Symbol,Int}(dictmap([x.name for x in domain.constants]))
    MixedLRNN2(domain, multiarg_predicates, nunanary_predicates, objtype2id, constmap, model_params, nothing, nothing, nothing)
end

isspecialized(ex::MixedLRNN2) = ex.obj2id !== nothing
hasgoal(ex::MixedLRNN2) = ex.init_state !== nothing || ex.goal_state !== nothing


function MixedLRNN2(domain, problem; embed_goal=true, kwargs...)
    ex = MixedLRNN2(domain; kwargs...)
    ex = specialize(ex, problem)
    embed_goal ? add_goalstate(ex, problem) : ex
end

function Base.show(io::IO, ex::MixedLRNN2)
    if !isspecialized(ex)
        print(io, "Unspecialized extractor for ", ex.domain.name, " (", length(ex.nunanary_predicates), ", ", length(ex.multiarg_predicates), ")")
    else
        g = hasgoal(ex) ? "with" : "without"
        print(io, "Specialized extractor ", g, " goal for ", ex.domain.name, " (", length(ex.nunanary_predicates), ", ", length(ex.multiarg_predicates), ", ", length(ex.obj2id), ")")
    end
end

"""
specialize(ex::MixedLRNN2{<:Nothing,<:Nothing}, problem)

initializes extractor for a given `problem` by initializing mapping 
from objects to id of vertices. Goals are not changed added to the 
extractor.
"""
function specialize(ex::MixedLRNN2, problem)
    obj2id = Dict(v.name => i for (i, v) in enumerate(problem.objects))
    for k in keys(ex.constmap)
        obj2id[k] = length(obj2id) + 1
    end
    MixedLRNN2(ex.domain, ex.multiarg_predicates, ex.nunanary_predicates, ex.objtype2id, ex.constmap, ex.model_params, obj2id, nothing, nothing)
end

function (ex::MixedLRNN2)(state::GenericState)
    prefix = (ex.goal_state !== nothing) ? :start : ((ex.init_state !== nothing) ? :goal : nothing)
    kb = encode_state(ex, state, prefix)
    addgoal(ex, kb)
end

function encode_state(ex::MixedLRNN2, state::GenericState, prefix=nothing)
    message_passes, residual = ex.model_params
    x = nunary_predicates(ex, state)
    kb = KnowledgeBase((; x1=x))
    n = size(x, 2)
    sₓ = :x1
    edge_structure = multi_predicates(ex, :x1, state, prefix)
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
nunary_predicates(ex::MixedLRNN2, state)

Create matrix with one column per object and encode by one-hot-encoding unary predicates 
and types of objects. Nunary predicates are encoded as properties of all objects.
"""
function nunary_predicates(ex::MixedLRNN2, state)
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

    for f in get_facts(state)
        length(get_args(f)) > 1 && continue
        v = 1
        if (f isa PDDL.Compound) && (f.name == :not)
            f = f.args[1]
            v = 0
        end
        a = get_args(f)
        pid = ex.nunanary_predicates[f.name]
        if length(a) == 1
            vid = ex.obj2id[get_args(f)[1].name]
            x[pid, vid] = v
        else
            length(a) == 0
            x[pid, :] .= v
        end
    end
    x
end

function group_facts(ex::MixedLRNN2, facts::Vector{<:Term})
    ps = [k => Int[] for k in ex.multiarg_predicates]
    occurences = Dict(ps)
    for (i, f) in enumerate(facts)
        f.name ∉ keys(occurences) && continue
        push!(occurences[f.name], i)
    end
    occurences
end

function group_facts_fast(ex::MixedLRNN2, facts::Vector{<:Term})
    occurences = falses(length(facts), length(ex.multiarg_predicates))
    for (i, f) in enumerate(facts)
        # f.name ∉ keys(occurences) && continue
        f.name ∉ ex.multiarg_predicates && continue
        col = findfirst(==(f.name), ex.multiarg_predicates)
        occurences[i, col] = true
    end
    _mapenumerate_tuple((col, k) -> k => (@view occurences[:, col]), ex.multiarg_predicates)
end

# this is better
function multi_predicates(ex::MixedLRNN2, kid::Symbol, state, prefix=nothing)
    # Then, we specify the predicates the dirty way
    ks = ex.multiarg_predicates
    facts = collect(get_facts(state))
    gr = group_facts_fast(ex, facts)
    xs = map(kii -> encode_predicates(ex, kii[1], facts[kii[2]], kid), gr)
    ns = isnothing(prefix) ? ks : _map_tuple(k -> Symbol(prefix, "_", k), ks)
    ProductNode(NamedTuple{ns}(xs))
end

function encode_predicates(ex::MixedLRNN2, pred_name::Symbol, preds, kid::Symbol)
    arity = length(ex.domain.predicates[pred_name].args)
    encode_predicates(ex, Val(arity), preds, kid)
    # encode_predicates_comp(ex, Val(arity), preds, kid)
end

function encode_predicates(ex::MixedLRNN2, arity::Val{N}, preds, kid::Symbol) where {N}
    eb = EdgeBuilder(N, length(preds), length(ex.obj2id))
    for p in preds
        edge = _map_tuple(i -> ex.obj2id[p.args[i].name], arity)
        push!(eb, edge)
    end
    construct(eb, kid)
end

function encode_predicates_comp(ex::MixedLRNN2, arity::Val{N}, preds, kid::Symbol) where {N}
    eb = EdgeBuilderComp(N, length(preds), length(ex.obj2id))
    for p in preds
        edge = _map_tuple(i -> ex.obj2id[p.args[i].name], arity)
        push!(eb, edge)
    end
    construct(eb, kid)
end

function add_goalstate(ex::MixedLRNN2, problem, goal=goalstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    @set ex.goal_state = encode_state(ex, goal, :goal)
end

function add_initstate(ex::MixedLRNN2, problem, start=initstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    @set ex.init_state = encode_state(ex, start, :start)
end

function addgoal(ex::MixedLRNN2Start, kb::KnowledgeBase)
    return (stack_hypergraphs(ex.init_state, kb))
end

function addgoal(ex::MixedLRNN2Goal, kb::KnowledgeBase)
    return (stack_hypergraphs(kb, ex.goal_state))
end

function addgoal(ex::MixedLRNN2NoGoal, kb::KnowledgeBase)
    return (kb)
end
