using PDDL: get_facts, get_args
"""
```julia
struct LRNN{DO,D,N,G}
	domain::DO
	multiarg_predicates::NTuple{N,Symbol}
	unary_predicates::Dict{Symbol,Int64}
	nullary_predicates::Dict{Symbol,Int64}
	objtype2id::Dict{Symbol,Int64}
	constmap::Dict{Symbol, Int64}
	model_params::NamedTuple{(:message_passes, :residual), Tuple{Int64, Symbol}}
	obj2id::D
	goal::G
end
```

LRNN (Lifted Relational Neural Network) represents a PDDL state as a hypergraph, whre 
- Each node is either an object or a contant
- n-ary predicate is a hyper-edge

The computational model is message-passing over hyper-graph, which is essentially 
a message passing over a bipartite graph, where left vertices corresponds to vertices
in hypergraph and right vertices corresponds to hyper-edges. There is an edge between
vertex corresponding to the hyper-edge and its vertices.

--- `multiarg_predicates` is a list of all n-ary predicates
--- `unary_predicates` maps unary predicates to an index in one-hot encoded vertex' properties 
--- `nullary_predicates` maps nullary predicates to an index in one-hot encoded vertex' properties 
--- `objtype2id` maps unary predicates to an index in one-hot encoded vertex' properties 
--- `constmap` maps constants to an index in one-hot encoded vertex' properties 
--- `model_params` some parameters of an algorithm constructing the message passing passes 
"""
struct LRNN{DO,D,N,MP,S,G}
    domain::DO
    multiarg_predicates::NTuple{N,Symbol}
    unary_predicates::Dict{Symbol,Int64}
    nullary_predicates::Dict{Symbol,Int64}
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol,Int64}
    model_params::MP
    obj2id::D
    init_state::S
    goal_state::G
    function LRNN(domain::DO, multiarg_predicates::NTuple{N,Symbol}, unary_predicates::Dict{Symbol,Int64}, nullary_predicates::Dict{Symbol,Int64},
        objtype2id::Dict{Symbol,Int64}, constmap::Dict{Symbol,Int64}, model_params::MP, obj2id::D, init::S, goal::G) where {DO,D,N,MP<:NamedTuple,S,G}
        @assert issubset((:message_passes, :residual), keys(model_params)) "Parameters of the model are not fully specified"
        @assert (init === nothing || goal === nothing) "Fixing init and goal state is bizzaare, as the extractor would always create a constant"
        new{DO,D,N,MP,S,G}(domain, multiarg_predicates, unary_predicates, nullary_predicates, objtype2id, constmap, model_params, obj2id, init, goal)
    end
end

LRNNNoGoal{DO,D,N,MP} = LRNN{DO,D,N,MP,Nothing,Nothing} where {DO,D,N,MP}
LRNNStart{DO,D,N,MP,S} = LRNN{DO,D,N,MP,S,Nothing} where {DO,D,N,MP,S<:KnowledgeBase}
LRNNGoal{DO,D,N,MP,S} = LRNN{DO,D,N,MP,Nothing,S} where {DO,D,N,MP,S<:KnowledgeBase}

function LRNN(domain; message_passes=2, residual=:linear, kwargs...)
    model_params = (; message_passes, residual)
    dictmap(x) = Dict(reverse.(enumerate(sort(x))))
    predicates = collect(domain.predicates)
    multiarg_predicates = tuple([kv[1] for kv in predicates if length(kv[2].args) > 1]...)
    unary_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) == 1])
    nullary_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) < 1])
    objtype2id = Dict(s => i + length(unary_predicates) for (i, s) in enumerate(collect(keys(domain.typetree))))
    constmap = Dict{Symbol,Int}(dictmap([x.name for x in domain.constants]))
    LRNN(domain, multiarg_predicates, unary_predicates, nullary_predicates, objtype2id, constmap, model_params, nothing, nothing, nothing)
end

isspecialized(ex::LRNN) = ex.obj2id !== nothing
hasgoal(ex::LRNN) = ex.init_state !== nothing || ex.goal_state !== nothing


function LRNN(domain, problem; embed_goal=true, kwargs...)
    ex = LRNN(domain; kwargs...)
    ex = specialize(ex, problem)
    embed_goal ? add_goalstate(ex, problem) : ex
end

function Base.show(io::IO, ex::LRNN)
    if !isspecialized(ex)
        print(io, "Unspecialized extractor for ", ex.domain.name, " (", length(ex.unary_predicates), ", ", length(ex.multiarg_predicates), ")")
    else
        g = hasgoal(ex) ? "with" : "without"
        print(io, "Specialized extractor ", g, " goal for ", ex.domain.name, " (", length(ex.unary_predicates), ", ", length(ex.multiarg_predicates), ", ", length(ex.obj2id), ")")
    end
end

"""
```julia
specialize(ex::LRNN, problem)
```

initializes extractor for a given `problem` by initializing mapping 
from objects to id of vertices. Goals are not changed added to the 
extractor.
"""
function specialize(ex::LRNN, problem)
    obj2id = Dict(v.name => i for (i, v) in enumerate(problem.objects))
    for k in keys(ex.constmap)
        obj2id[k] = length(obj2id) + 1
    end
    LRNN(ex.domain, ex.multiarg_predicates, ex.unary_predicates, ex.nullary_predicates, ex.objtype2id, ex.constmap, ex.model_params, obj2id, nothing, nothing)
end

function (ex::LRNN)(state::GenericState)
    prefix = (ex.goal_state !== nothing) ? :start : ((ex.init_state !== nothing) ? :goal : nothing)
    kb = encode_state(ex, state, prefix)
    addgoal(ex, kb)
end

function encode_state(ex::LRNN, state::GenericState, prefix=nothing)
    message_passes, residual = ex.model_params
    # x = zeros(Float32, 1, length(ex.obj2id))
    x = ones(Float32, 1, length(ex.obj2id))
    kb = KnowledgeBase((; x1=x))
    n = size(x, 2)
    sₓ = :x1
    if any(!isempty, (ex.multiarg_predicates, ex.unary_predicates, ex.nullary_predicates))
        for _ in 1:message_passes
            input_to_gnn = last(keys(kb))
            ds = encode_predicates(ex, input_to_gnn, state, prefix)
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
add_residual_layer(kb::KnowledgeBase, inputs::Tuple{Symbol}, n::Int)

adds a residual layer mixing `inputs` in `kb` KnowledgeBase over `n` items
"""
function add_residual_layer(kb::KnowledgeBase, inputs::NTuple{N,Symbol}, n::Int, prefix="res") where {N}
    childs = map(s -> ArrayNode(KBEntry(s, 1:n)), inputs)
    ds = ProductNode(childs)
    append(kb, layer_name(kb, prefix), ds)
end

"""
layer_name(kb::KnowledgeBase, prefix)

create a unique name of the layer for KnowledgeBase `kb`
"""
layer_name(kb::KnowledgeBase{KS,<:Any}, prefix) where {KS} = Symbol(prefix,"_",length(KS)+1)

"""
encode_predicates(ex::LRNN, kid::Symbol, state, prefix=nothing)

function for encoding n-ary predicates into knowledge base
"""
function encode_predicates(ex::LRNN, kid::Symbol, state, prefix=nothing)
    ksm = ex.multiarg_predicates
    xsm = map(ksm) do k
        preds = filter(f -> f.name == k, get_facts(state))
        encode_multiarg_predicates(ex, k, preds, kid)
    end

    ksu = ex.unary_predicates
    xsu = map(Tuple(ksu)) do (k, _)
        preds = filter(f -> f.name == k, get_facts(state))
        encode_unary_predicates(ex, preds, kid)
    end

    ksn = ex.nullary_predicates
    xsn = map(Tuple(ksn)) do (_, _)
        encode_nullary_predicates(ex, kid)
    end

    ks = (ksm..., collect(keys(ksu))..., collect(keys(ksn))...)
    xs = (xsm..., xsu..., xsn...)

    ns = isnothing(prefix) ? ks : tuple([Symbol(prefix, "_", k) for k in ks]...)

    ProductNode(NamedTuple{ns}(xs))
end

"""
```julia
encode_multiarg_predicates(ex::LRNN, pname::Symbol, preds, kid::Symbol)
```

encodes predicates with arity greater than 2 into bag node
"""
function encode_multiarg_predicates(ex::LRNN, pname::Symbol, preds, kid::Symbol)
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

function encode_unary_predicates(ex::LRNN, preds, kid::Symbol)
    obj2id = ex.obj2id

    xs = ArrayNode(KBEntry(kid, [obj[2] for obj in obj2id]))

    ar = zeros(Bool, length(obj2id))
    for pred in preds
        ar[obj2id[pred.args[1].name]] = 1
    end

    mask = BitVector(ar)

    MaskedNode(xs, mask)
end

function encode_nullary_predicates(ex::LRNN, kid::Symbol)
    obj2id = ex.obj2id

    xs = ArrayNode(KBEntry(kid, [obj[2] for obj in obj2id]))
    MaskedNode(xs)
end

function add_goalstate(ex::LRNN, problem, goal=goalstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    exg = encode_state(ex, goal, :goal)
    LRNN(ex.domain, ex.multiarg_predicates, ex.unary_predicates, ex.nullary_predicates, ex.objtype2id, ex.constmap, ex.model_params, ex.obj2id, nothing, exg)
end

function add_initstate(ex::LRNN, problem, start=initstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    exg = encode_state(ex, start, :start)
    LRNN(ex.domain, ex.multiarg_predicates, ex.unary_predicates, ex.nullary_predicates, ex.objtype2id, ex.constmap, ex.model_params, ex.obj2id, exg, nothing)
end

function addgoal(ex::LRNNStart, kb::KnowledgeBase)
    return (stack_hypergraphs(ex.init_state, kb))
end

function addgoal(ex::LRNNGoal, kb::KnowledgeBase)
    return (stack_hypergraphs(kb, ex.goal_state))
end

function addgoal(ex::LRNNNoGoal, kb::KnowledgeBase)
    return (kb)
end

function stack_hypergraphs(kb1::KnowledgeBase{KX,V1}, kb2::KnowledgeBase{KX,V2}) where {KX,V1,V2}
    x = vcat(kb1[:x1], kb2[:x1])
    gp = map(KX[2:end-1]) do k
        if _isstackable(kb1[k], kb2[k])
            ProductNode(merge(kb1[k].data, kb2[k].data))
        else
            kb1[k]
        end
    end
    KnowledgeBase(NamedTuple{KX}(tuple(x, gp..., kb1.kb[end])))
end

"""
Checks if two ProductNodes should be stacked on top of each other. 
"""
function _isstackable(ds1::ProductNode{<:NamedTuple}, ds2::ProductNode{<:NamedTuple})
    return numobs(ds1) === numobs(ds2)
end
_isstackable(ds1, ds2) = false
