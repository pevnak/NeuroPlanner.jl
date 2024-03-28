using PDDL: get_facts, get_args
"""
struct ObjectPair{DO,D,N,MP,S,G}
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

struct ObjectPair{DO,D,MP,DV,DT,V,S,G}
    domain::DO
    multiarg_predicates::Dict{Symbol,Int64}
    unary_predicates::Dict{Symbol,Int64}
    nullary_predicates::Dict{Symbol,Int64}
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol,Int64}
    model_params::MP
    obj2id::D
    obj2pid::DV
    pair2id::DT
    pairs::V
    init_state::S
    goal_state::G
    function ObjectPair(domain::DO, multiarg_predicates::Dict{Symbol,Int64}, unary_predicates::Dict{Symbol,Int64}, nullary_predicates::Dict{Symbol,Int64},
        objtype2id::Dict{Symbol,Int64}, constmap::Dict{Symbol,Int64}, model_params::MP, obj2id::D, obj2pid::DV, pair2id::DT,
        pairs::V, init::S, goal::G) where {DO,D,MP<:NamedTuple,DV,DT,V,S,G}

        @assert issubset((:message_passes, :residual), keys(model_params)) "Parameters of the model are not fully specified"
        @assert (init === nothing || goal === nothing) "Fixing init and goal state is bizzaare, as the extractor would always create a constant"
        new{DO,D,MP,DV,DT,V,S,G}(domain, multiarg_predicates, unary_predicates, nullary_predicates, objtype2id, constmap, model_params,
            obj2id, obj2pid, pair2id, pairs, init, goal)
    end
end


ObjectPairNoGoal{DO,D,MP,DV,DT,V} = ObjectPair{DO,D,MP,DV,DT,V,Nothing,Nothing} where {DO,D,MP,DV,DT,V}
ObjectPairStart{DO,D,MP,DV,DT,V,S} = ObjectPair{DO,D,MP,DV,DT,V,S,Nothing} where {DO,D,MP,DV,DT,V,S<:KnowledgeBase}
ObjectPairGoal{DO,D,MP,DV,DT,V,S} = ObjectPair{DO,D,MP,DV,DT,V,Nothing,S} where {DO,D,MP,DV,DT,V,S<:KnowledgeBase}

function ObjectPair(domain; message_passes=2, residual=:linear, kwargs...)
    model_params = (; message_passes, residual)
    dictmap(x) = Dict(reverse.(enumerate(sort(x))))
    predicates = collect(domain.predicates)
    multiarg_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) > 1])
    unary_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) == 1])
    nullary_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) < 1])
    objtype2id = Dict(s => i + length(unary_predicates) for (i, s) in enumerate(collect(keys(domain.typetree))))
    constmap = Dict{Symbol,Int}(dictmap([x.name for x in domain.constants]))
    ObjectPair(domain, multiarg_predicates, unary_predicates, nullary_predicates, objtype2id, constmap, model_params,
        nothing, nothing, nothing, nothing, nothing, nothing)
end

isspecialized(ex::ObjectPair) = ex.obj2id !== nothing
hasgoal(ex::ObjectPair) = ex.goal_state !== nothing

function ObjectPair(domain, problem; embed_goal=true, kwargs...)
    ex = ObjectPair(domain; kwargs...)
    ex = specialize(ex, problem)
    embed_goal ? add_goalstate(ex, problem) : ex
end

function Base.show(io::IO, ex::ObjectPair)
    if !isspecialized(ex)
        print(io, "Unspecialized extractor for ", ex.domain.name,
            " (", length(ex.nullary_predicates), ", ", length(ex.unary_predicates), ", ", length(ex.multiarg_predicates), ")")
    else
        g = hasgoal(ex) ? "with" : "without"
        print(io, "Specialized extractor ", g, " goal for ", ex.domain.name,
            " (", length(ex.nullary_predicates) + 2 * (length(ex.unary_predicates) + length(ex.objtype2id) + length(ex.constmap)) + length(ex.multiarg_predicates),
            ", ", length(ex.pairs), ")")
    end
end


"""
specialize(ex::ObjectPair{<:Nothing,<:Nothing}, problem)

initializes extractor for a given `problem` by initializing mapping 
from objects to id of vertices. Goals are not changed added to the 
extractor.
"""
function specialize(ex::ObjectPair, problem)
    obj2id = Dict(v.name => i for (i, v) in enumerate(problem.objects))
    obj2idv = [(v.name, i) for (i, v) in enumerate(problem.objects)]

    for k in keys(ex.constmap)
        obj2id[k] = length(obj2id) + 1
        push!(obj2idv, (k, length(obj2idv) + 1))
    end

    obj2pid = Dict(v[1] => Tuple{Int64,Int64}[] for v in obj2idv)
    pair2id = Dict{Tuple{Symbol,Symbol},Int32}()

    offset = -length(obj2idv)

    pairs = map(obj2idv) do (name, id)
        # global offset
        offset += length(obj2idv) + 1 - id
        p = Tuple{Symbol,Symbol}[]
        for i in id:length(obj2idv)
            push!(obj2pid[name], (offset + i, 1))
            push!(obj2pid[obj2idv[i][1]], (offset + i, 2))
            push!(p, (name, obj2idv[i][1]))
            pair2id[(name, obj2idv[i][1])] = offset + i
        end
        p
    end 
    pairs = reduce(vcat, pairs)

    ObjectPair(ex.domain, ex.multiarg_predicates, ex.unary_predicates, ex.nullary_predicates, ex.objtype2id,
        ex.constmap, ex.model_params, obj2id, obj2pid, pair2id, pairs, nothing, nothing)
end

function (ex::ObjectPair)(state::GenericState)
    prefix = (ex.goal_state !== nothing) ? :start : ((ex.init_state !== nothing) ? :goal : nothing)
    kb = encode_state(ex, state, prefix)
    addgoal(ex, kb)
end

function encode_state(ex::ObjectPair, state::GenericState, prefix=nothing)
    message_passes, residual = ex.model_params
    x = feature_vectors(ex, state)
    kb = KnowledgeBase((; x1=x))
    n = size(x, 2)
    sₓ = :x1
    if !isempty(ex.multiarg_predicates)
        for i in 1:message_passes
            input_to_gnn = last(keys(kb))
            ds = encode_edges(ex, input_to_gnn, state, prefix)
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
feature_vectors(ex::ObjectPair, state) 

Creates a matrix with one column per pairs of objects and encode features by one-hot-encoding
"""
function feature_vectors(ex::ObjectPair, state)
    idim = length(ex.nullary_predicates) + 2 * (length(ex.unary_predicates) + length(ex.objtype2id) + length(ex.constmap)) + length(ex.multiarg_predicates)
    x = zeros(Float32, idim, length(ex.pairs))

    # length of feature vector of one object in a pair
    obj_length = length(ex.unary_predicates) + length(ex.objtype2id) + length(ex.constmap)

    # nullary_predicates
    for (_, i) in ex.nullary_predicates
        x[i, :] .= 1
    end

    # encode unary predicates
    offset = length(ex.nullary_predicates)
    for f in get_facts(state)
        length(get_args(f)) != 1 && continue
        (f isa PDDL.Compound) && (f.name == :not) && continue

        pred_id = ex.unary_predicates[f.name]
        vids = ex.obj2pid[only(get_args(f)).name]
        for (vid, pos) in vids
            x[offset+(pos-1)*obj_length+pred_id, vid] = 1
        end
    end

    # encode types of objects
    offset = length(ex.nullary_predicates)
    for s in state.types
        i = ex.objtype2id[s.name]
        js = ex.obj2pid[only(s.args).name]
        for (j, pos) in js
            x[offset+(pos-1)*obj_length+i, j] = 1
        end
    end

    # encode constants
    offset = length(ex.nullary_predicates) + length(ex.unary_predicates) + length(ex.objtype2id)
    for (k, i) in ex.constmap
        js = ex.obj2pid[k]
        for (j, pos) in js
            x[offset+(pos-1)*obj_length+i, j] = 1
        end
    end


    # encode predicate relatedness
    offset = length(ex.nullary_predicates) + 2 * obj_length
    for f in get_facts(state)
        length(get_args(f)) < 2 && continue
        (f isa PDDL.Compound) && (f.name == :not) && continue

        pred_id = ex.multiarg_predicates[f.name]
        is = [ex.obj2id[o.name] for o in get_args(f)]

        for (i, id) in enumerate(is)
            for j in i+1:length(is)
                # ordering ids of objects for formula to work
                id1, id2 = id < is[j] ? (id, is[j]) : (is[j], id)
                # formula for getting pair id
                pid = Int(((length(ex.obj2id) + (length(ex.obj2id) - (id1 - 1) + 1)) * (id1 - 1) / 2)) + 1 + (id2 - id1)
                x[offset+pred_id, pid] = 1
            end
        end
    end

    x
end

"""
function encode_edges(ex::ObjectPair, kid::Symbol, state, prefix=nothing)

Creates ProductNode of named BagNodes each representing one labeled edge in multigraph.
"""
function encode_edges(ex::ObjectPair, kid::Symbol, state, prefix=nothing)
    name, edges = encode_E_edges(ex, kid; prefix=prefix)
    ns, xs = multi_predicates(ex, kid, state, prefix)

    ProductNode(NamedTuple{(name, ns...)}((edges, xs...)))
end

"""
function encode_E_edges(ex::ObjectPair, kid::Symbol; sym=:edge, prefix=nothing)

Encodes `E` Edges, which connect two pairs of object if and only if size of conjunction of their objects is equal to 1.
"""
function encode_E_edges(ex::ObjectPair, kid::Symbol; sym=:edge, prefix=nothing)
    xs = map(collect(keys(ex.obj2id))) do obj
        pairs = ex.obj2pid[obj]
        edges = Tuple{Int64,Int64}[]
        for i in eachindex(pairs)
            pairs[i][2] != 1 && continue
            pid = pairs[i][1]
            es = [(pid, pairs[j][1]) for j in i+1:length(pairs) if pid != pairs[j][1]]
            push!(edges, es...)
        end
        edges
    end |> (arrays -> vcat(arrays...))

    x = map(1:2) do i
        ArrayNode(KBEntry(kid, map(p -> p[i], xs)))
    end |> (an -> ProductNode(tuple(an...)))

    bags = [Int64[] for _ in 1:length(ex.pairs)]
    for (i, f) in enumerate(xs)
        for j in 1:2
            push!(bags[f[j]], i)
        end
    end

    name = isnothing(prefix) ? sym : Symbol(prefix, "_", sym)
    (name, BagNode(x, ScatteredBags(bags)))
end

"""
function multi_predicates(ex::ObjectPair, kid::Symbol, state, prefix=nothing)

Encodes predicates with arity greater than 1 to edges that connect pairs of objects whose objects are related by given predicate.
"""
function multi_predicates(ex::ObjectPair, kid::Symbol, state, prefix=nothing)
    # Then, we specify the predicates the dirty way
    ks = tuple(collect(keys(ex.multiarg_predicates))...)
    xs = map(ks) do k
        preds = filter(f -> f.name == k, get_facts(state))
        encode_predicates(ex, k, preds, kid)
    end
    ns = isnothing(prefix) ? ks : tuple([Symbol(prefix, "_", k) for k in ks]...)
    (ns, xs)
end


"""
function encode_predicates(ex::ObjectPair, pname::Symbol, preds, kid::Symbol)
Encode predicates for binary relations.

This function encodes predicates for binary relations.

# Arguments
- `ex::ObjectPair`: An extractor representing pddl architecture.
- `pname::Symbol`: The symbol representing the predicate name.
- `preds`: The predicates.
- `kid::Symbol`: The symbol representing the previous entry in KB.

# Returns
- `BagNode`: A bag node containing the encoded predicates.
"""
function encode_predicates(ex::ObjectPair, pname::Symbol, preds, kid::Symbol)
    xs = map(collect(preds)) do f
        xs = unique(x[1] for x in ex.obj2pid[f.args[1].name])
        ys = unique(x[1] for x in ex.obj2pid[f.args[2].name])

        [(x, y) for x in xs for y in ys if x != y]
    end 
    xs = reduce(vcat, xs)

    x = map(1:2) do i
        ArrayNode(KBEntry(kid, map(p -> p[i], xs)))
    end
    x = ProductNode(tuple(x...))


    bags = [Int64[] for _ in 1:length(ex.pairs)]

    for (i, f) in enumerate(xs)
        for j in 1:2
            push!(bags[f[j]], i)
        end
    end

    BagNode(x, ScatteredBags(bags))
end


function add_goalstate(ex::ObjectPair, problem, goal=goalstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    exg = encode_state(ex, goal, :goal)
    ObjectPair(ex.domain, ex.multiarg_predicates, ex.unary_predicates, ex.nullary_predicates, ex.objtype2id, ex.constmap,
        ex.model_params, ex.obj2id, ex.obj2pid, ex.pair2id, ex.pairs, nothing, exg)
end

function add_initstate(ex::ObjectPair, problem, start=initstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    exg = encode_state(ex, start, :start)
    ObjectPair(ex.domain, ex.multiarg_predicates, ex.unary_predicates, ex.nullary_predicates, ex.objtype2id, ex.constmap,
        ex.model_params, ex.obj2id, ex.obj2pid, ex.pair2id, ex.pairs, exg, nothing)
end

function addgoal(ex::ObjectPairStart, kb::KnowledgeBase)
    return (stack_hypergraphs(ex.init_state, kb))
end

function addgoal(ex::ObjectPairGoal, kb::KnowledgeBase)
    return (stack_hypergraphs(kb, ex.goal_state))
end

function addgoal(ex::ObjectPairNoGoal, kb::KnowledgeBase)
    return (kb)
end
