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

struct ObjectPair{DO,D,MP,DV,DUV,DT,V,S,G}
    domain::DO
    multiarg_predicates::Dict{Symbol,Int64}
    unary_predicates::Dict{Symbol,Int64}
    nullary_predicates::Dict{Symbol,Int64}
    objtype2id::Dict{Symbol,Int64}
    constmap::Dict{Symbol,Int64}
    model_params::MP
    obj2id::D
    obj2pid::DV
    obj2upid::DUV
    pair2pid::DT
    pairs::V
    init_state::S
    goal_state::G
    function ObjectPair(domain::DO, multiarg_predicates::Dict{Symbol,Int64}, unary_predicates::Dict{Symbol,Int64}, nullary_predicates::Dict{Symbol,Int64},
        objtype2id::Dict{Symbol,Int64}, constmap::Dict{Symbol,Int64}, model_params::MP, obj2id::D, obj2pid::DV, obj2upid::DUV, pair2pid::DT,
        pairs::V, init::S, goal::G) where {DO,D,MP<:NamedTuple,DV,DUV,DT,V,S,G}

        @assert issubset((:message_passes, :residual), keys(model_params)) "Parameters of the model are not fully specified"
        @assert (init === nothing || goal === nothing) "Fixing init and goal state is bizzaare, as the extractor would always create a constant"
        new{DO,D,MP,DV,DUV,DT,V,S,G}(domain, multiarg_predicates, unary_predicates, nullary_predicates, objtype2id, constmap, model_params,
            obj2id, obj2pid, obj2upid, pair2pid, pairs, init, goal)
    end
end


ObjectPairNoGoal{DO,D,MP,DV,DUV,DT,V} = ObjectPair{DO,D,MP,DV,DUV,DT,V,Nothing,Nothing} where {DO,D,MP,DV,DUV,DT,V}
ObjectPairStart{DO,D,MP,DV,DUV,DT,V,S} = ObjectPair{DO,D,MP,DV,DUV,DT,V,S,Nothing} where {DO,D,MP,DV,DUV,DT,V,S<:KnowledgeBase}
ObjectPairGoal{DO,D,MP,DV,DUV,DT,V,S} = ObjectPair{DO,D,MP,DV,DUV,DT,V,Nothing,S} where {DO,D,MP,DV,DUV,DT,V,S<:KnowledgeBase}

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
        nothing, nothing, nothing, nothing, nothing, nothing, nothing)
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

    obj2pid = Dict(v[1] => Vector{Tuple{Int64,Int64}}(undef, length(obj2id) + 1) for v in obj2idv)
    obj2upid = Dict(v[1] => Vector{Int64}(undef, length(obj2id)) for v in obj2idv)
    indices = fill(1, length(obj2id))
    uindices = fill(1, length(obj2id))
    pair2pid = Dict{Tuple{Symbol,Symbol},Int64}()

    pairs_count = Int((1 + length(obj2idv)) * length(obj2idv) / 2)

    pairs = [(Symbol("a_$i"), Symbol("b_$i")) for i in 1:pairs_count]
    offset = -length(obj2idv)

    for (name, id) in obj2idv
        # global offset
        offset += length(obj2idv) + 1 - id
        for i in id:length(obj2idv)
            obj2pid[name][indices[obj2id[name]]] = (offset + i, 1)
            indices[obj2id[name]] += 1

            obj2upid[name][uindices[obj2id[name]]] = offset + i
            uindices[obj2id[name]] += 1

            o₂ = obj2idv[i][1]

            obj2pid[o₂][indices[obj2id[o₂]]] = (offset + i, 2)
            indices[obj2id[o₂]] += 1

            if (i != id)
                obj2upid[o₂][uindices[obj2id[o₂]]] = offset + i
                uindices[obj2id[o₂]] += 1
            end

            pairs[offset+i] = (name, o₂)
            pair2pid[(name, o₂)] = offset + i
        end
    end

    ObjectPair(ex.domain, ex.multiarg_predicates, ex.unary_predicates, ex.nullary_predicates, ex.objtype2id,
        ex.constmap, ex.model_params, obj2id, obj2pid, obj2upid, pair2pid, pairs, nothing, nothing)
end

function (ex::ObjectPair)(state::GenericState)
    prefix = (ex.goal_state !== nothing) ? :start : ((ex.init_state !== nothing) ? :goal : nothing)
    kb = encode_state(ex, state, prefix)
    addgoal(ex, kb)
end

function encode_state(ex::ObjectPair, state::GenericState, prefix=nothing)
    message_passes, residual = ex.model_params
    grouped_facts = group_facts(ex, collect(PDDL.get_facts(state)))
    x = feature_vectors(ex, state)
    kb = KnowledgeBase((; x1=x))
    n = size(x, 2)
    sₓ = :x1

    edge_structure = encode_edges(ex, :x1, grouped_facts, prefix)
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
function encode_edges(ex::ObjectPair, kid::Symbol, grouped_facts, prefix=nothing)
    name, edges_bn, E_eb = encode_E_edges(ex, kid; prefix=prefix)

    pids, bags = prepare_pids(ex, E_eb)

    ns, xs = multi_predicates(ex, kid, grouped_facts, pids, bags, prefix)

    ProductNode(NamedTuple{(name, ns...)}((edges_bn, xs...)))
end

function prepare_pids(ex::ObjectPair, eb::EdgeBuilder)
    counts = fill(0, length(ex.pairs))
    pids = Vector{Int}(undef, eb.max_edges)

    for pid in eb.indices[1:eb.max_edges]
        counts[pid] += 1
    end

    ends = cumsum(counts)
    start = ends .- (counts .- 1)
    bags = map((x, y) -> x:y, start, ends)

    for (i, pid) in enumerate(eb.indices[1:eb.max_edges])
        pids[start[pid]] = eb.indices[eb.max_edges+i]
        start[pid] += 1
    end
    return (pids, bags)
end

"""
function encode_E_edges(ex::ObjectPair, kid::Symbol; sym=:edge, prefix=nothing)

Encodes `E` Edges, which connect two pairs of object if and only if size of conjunction of their objects is equal to 1.
"""
function encode_E_edges(ex::ObjectPair, kid::Symbol; sym=:edge, prefix=nothing)
    n = Int((1 + (length(ex.obj2id) - 1)) * (length(ex.obj2id) - 1) / 2) * length(ex.obj2pid)
    eb = EdgeBuilder(2, n, length(ex.pairs))

    for pairs in values(ex.obj2upid)
        for i in eachindex(pairs)
            pidᵢ = pairs[i]
            for j in i+1:length(pairs)
                pidⱼ = pairs[j]
                push!(eb, (pidᵢ, pidⱼ))
            end
        end
    end

    name = isnothing(prefix) ? sym : Symbol(prefix, "_", sym)
    (name, construct(eb, kid), eb)
end


function group_facts(ex::ObjectPair, facts::Vector{<:Term})
    multiarg_predicates = tuple(keys(ex.multiarg_predicates)...)
    occurences = falses(length(facts), length(ex.multiarg_predicates))
    for (i, f) in enumerate(facts)
        col = _inlined_search(f.name, multiarg_predicates)
        col == -1 && continue
        occurences[i, col] = true
    end

    _mapenumerate_tuple(multiarg_predicates) do col, k
        N = length(ex.domain.predicates[k].args)
        k => factargs2id(ex, facts, (@view occurences[:, col]), Val(N))
    end
end

function factargs2id(ex::ObjectPair, facts, mask, arity::Val{N}) where {N}
    # d = ex.obj2id
    o = Vector{NTuple{N,Symbol}}(undef, sum(mask))
    index = 1
    for i in 1:length(mask)
        mask[i] || continue
        p = facts[i]
        # o[index] = _map_tuple(j -> d[p.args[j].name], arity)
        o[index] = _map_tuple(j -> p.args[j].name, arity)
        index += 1
    end
    o
end


"""
function multi_predicates(ex::ObjectPair, kid::Symbol, state, prefix=nothing)

Encodes predicates with arity greater than 1 to edges that connect pairs of objects whose objects are related by given predicate.
"""
function multi_predicates(ex::ObjectPair, kid::Symbol, grouped_facts, pids::Vector{Int}, bags::Vector{UnitRange{Int64}}, prefix=nothing)
    # Then, we specify the predicates the dirty way
    ks = tuple(collect(keys(ex.multiarg_predicates))...)
    xs = map(kii -> encode_predicates(ex, kii[2], pids, bags, kid), grouped_facts)
    ns = isnothing(prefix) ? ks : _map_tuple(k -> Symbol(prefix, "_", k), ks)
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

function encode_predicates(ex::ObjectPair, preds, pids::Vector{Int}, bags::Vector{UnitRange{Int}}, kid::Symbol)
    max_length = length(pids) * length(preds)
    eb = EdgeBuilder(2, max_length, length(ex.pairs))

    for f in preds
        xs = ex.obj2upid[f[1]]
        ys = ex.obj2upid[f[2]]
        for (x, y) in Iterators.product(xs, ys)
            x == y && continue
            y ∈ view(pids, bags[x]) || x ∈ view(pids, bags[y]) || continue
            push!(eb, (x, y))
        end
    end
    construct(eb, kid)
end


function add_goalstate(ex::ObjectPair, problem, goal=goalstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    @set ex.goal_state = encode_state(ex, goal, :goal)
end

function add_initstate(ex::ObjectPair, problem, start=initstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    @set ex.init_state = encode_state(ex, start, :start)
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
