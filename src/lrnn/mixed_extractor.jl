using PDDL: get_facts, get_args
"""
struct MixedLRNN{DO,D,N,G}
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
struct MixedLRNN{DO,D,N,MP,S,G}
    domain::DO
    multiarg_predicates::NTuple{N,Symbol}
    nunanary_predicates::Dict{Symbol,Int64}
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol,Int64}
    model_params::MP
    obj2id::D
    init_state::S
    goal_state::G
    function MixedLRNN(domain::DO, multiarg_predicates::NTuple{N,Symbol}, nunanary_predicates::Dict{Symbol,Int64}, objtype2id::Dict{Symbol,Int64}, constmap::Dict{Symbol,Int64}, model_params::MP, obj2id::D, init::S, goal::G) where {DO,D,N,MP<:NamedTuple,S,G}
        @assert issubset((:message_passes, :residual), keys(model_params)) "Parameters of the model are not fully specified"
        @assert (init === nothing || goal === nothing) "Fixing init and goal state is bizzaare, as the extractor would always create a constant"
        new{DO,D,N,MP,S,G}(domain, multiarg_predicates, nunanary_predicates, objtype2id, constmap, model_params, obj2id, init, goal)
    end
end

MixedLRNNNoGoal{DO,D,N,MP} = MixedLRNN{DO,D,N,MP,Nothing,Nothing} where {DO,D,N,MP}
MixedLRNNStart{DO,D,N,MP,S} = MixedLRNN{DO,D,N,MP,S,Nothing} where {DO,D,N,MP,S<:KnowledgeBase}
MixedLRNNGoal{DO,D,N,MP,S} = MixedLRNN{DO,D,N,MP,Nothing,S} where {DO,D,N,MP,S<:KnowledgeBase}

function MixedLRNN(domain; message_passes=2, residual=:linear, kwargs...)
    model_params = (; message_passes, residual)
    dictmap(x) = Dict(reverse.(enumerate(sort(x))))
    predicates = collect(domain.predicates)
    multiarg_predicates = tuple([kv[1] for kv in predicates if length(kv[2].args) > 1]...)
    nunanary_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) ≤ 1])
    objtype2id = Dict(s => i + length(nunanary_predicates) for (i, s) in enumerate(collect(keys(domain.typetree))))
    constmap = Dict{Symbol,Int}(dictmap([x.name for x in domain.constants]))
    MixedLRNN(domain, multiarg_predicates, nunanary_predicates, objtype2id, constmap, model_params, nothing, nothing, nothing)
end

isspecialized(ex::MixedLRNN) = ex.obj2id !== nothing
hasgoal(ex::MixedLRNN) = ex.init_state !== nothing || ex.goal_state !== nothing


function MixedLRNN(domain, problem; embed_goal=true, kwargs...)
    ex = MixedLRNN(domain; kwargs...)
    ex = specialize(ex, problem)
    embed_goal ? add_goalstate(ex, problem) : ex
end

function Base.show(io::IO, ex::MixedLRNN)
    if !isspecialized(ex)
        print(io, "Unspecialized extractor for ", ex.domain.name, " (", length(ex.nunanary_predicates), ", ", length(ex.multiarg_predicates), ")")
    else
        g = hasgoal(ex) ? "with" : "without"
        print(io, "Specialized extractor ", g, " goal for ", ex.domain.name, " (", length(ex.nunanary_predicates), ", ", length(ex.multiarg_predicates), ", ", length(ex.obj2id), ")")
    end
end

"""
specialize(ex::MixedLRNN{<:Nothing,<:Nothing}, problem)

initializes extractor for a given `problem` by initializing mapping 
from objects to id of vertices. Goals are not changed added to the 
extractor.
"""
function specialize(ex::MixedLRNN, problem)
    obj2id = Dict(v.name => i for (i, v) in enumerate(problem.objects))
    for k in keys(ex.constmap)
        obj2id[k] = length(obj2id) + 1
    end
    MixedLRNN(ex.domain, ex.multiarg_predicates, ex.nunanary_predicates, ex.objtype2id, ex.constmap, ex.model_params, obj2id, nothing, nothing)
end

function (ex::MixedLRNN)(state::GenericState)
    prefix = (ex.goal_state !== nothing) ? :start : ((ex.init_state !== nothing) ? :goal : nothing)
    kb = encode_state(ex, state, prefix)
    addgoal(ex, kb)
end

function encode_state(ex::MixedLRNN, state::GenericState, prefix=nothing)
    message_passes, residual = ex.model_params
    x = nunary_predicates(ex, state)
    kb = KnowledgeBase((; x1=x))
    n = size(x, 2)
    sₓ = :x1
    if !isempty(ex.multiarg_predicates)
        for i in 1:message_passes
            input_to_gnn = last(keys(kb))
            ds = multi_predicates(ex, input_to_gnn, state, prefix)
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
nunary_predicates(ex::MixedLRNN, state)

Create matrix with one column per object and encode by one-hot-encoding unary predicates 
and types of objects. Nunary predicates are encoded as properties of all objects.
"""
function nunary_predicates(ex::MixedLRNN, state)
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

    for f in get_facts(state)
        length(get_args(f)) > 1 && continue
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

function multi_predicates(ex::MixedLRNN, kid::Symbol, state, prefix=nothing)
    # Then, we specify the predicates the dirty way
    ks = ex.multiarg_predicates
    xs = map(ks) do k
        preds = filter(f -> f.name == k, get_facts(state))
        encode_predicates(ex, k, preds, kid)
    end
    ns = isnothing(prefix) ? ks : _map_tuple(k -> Symbol(prefix, "_", k), ks)
    ProductNode(NamedTuple{ns}(xs))
end

function encode_predicates(ex::MixedLRNN, pname::Symbol, preds, kid::Symbol)
    p = ex.domain.predicates[pname]
    obj2id = ex.obj2id
    constmap = ex.constmap
    xs = map(1:length(p.args)) do i
        syms = [f.args[i].name for f in preds]
        ArrayNode(KBEntry(kid, [obj2id[s] for s in syms]))
    end
    x = ProductNode(tuple(xs...))

    bags = [Int[] for _ in 1:length(obj2id)]
    for (j, f) in enumerate(preds)
        for a in f.args
            a.name ∉ keys(obj2id) && continue
            push!(bags[obj2id[a.name]], j)
        end
    end
    BagNode(x, ScatteredBags(bags))
end

# function encode_predicates_2(ex::MixedLRNN, pname::Symbol, preds, kid::Symbol)
#     p = ex.domain.predicates[pname]

#     n = length(preds)
#     indices = Vector{Int}(undef, n * length(p.args))
#     counts = fill(0, length(ex.obj2id))

#     obj2id = ex.obj2id
#     constmap = ex.constmap
#     xs = map(1:length(p.args)) do i
#         syms = [f.args[i].name for f in preds]
#         ArrayNode(KBEntry(kid, [obj2id[s] for s in syms]))
#     end
#     x = ProductNode(tuple(xs...))

#     for (i, f) in enumerate(preds)
#         for (j, a) in enumerate(f.args)
#             a.name ∉ keys(obj2id) && continue
#             oi = obj2id[a.name]
#             counts[oi] += 1
#             indices[(j-1)*n+j] = oi
#         end
#     end
#     BagNode(x, CompressedBags(indices, counts, n))
# end


function add_goalstate(ex::MixedLRNN, problem, goal=goalstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    exg = encode_state(ex, goal, :goal)
    MixedLRNN(ex.domain, ex.multiarg_predicates, ex.nunanary_predicates, ex.objtype2id, ex.constmap, ex.model_params, ex.obj2id, nothing, exg)
end

function add_initstate(ex::MixedLRNN, problem, start=initstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    exg = encode_state(ex, start, :start)
    MixedLRNN(ex.domain, ex.multiarg_predicates, ex.nunanary_predicates, ex.objtype2id, ex.constmap, ex.model_params, ex.obj2id, exg, nothing)
end

function addgoal(ex::MixedLRNNStart, kb::KnowledgeBase)
    return (stack_hypergraphs(ex.init_state, kb))
end

function addgoal(ex::MixedLRNNGoal, kb::KnowledgeBase)
    return (stack_hypergraphs(kb, ex.goal_state))
end

function addgoal(ex::MixedLRNNNoGoal, kb::KnowledgeBase)
    return (kb)
end
