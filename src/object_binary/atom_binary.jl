using PDDL: get_facts, get_args
"""
struct AtomBinary{DO,D,N,G}
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
struct AtomBinary{DO,MP,D,S,G}
    domain::DO
    max_arity::Int
    constmap::Dict{Symbol,Int}
    model_params::MP
    obj2id::D
    init_state::S
    goal_state::G
    function AtomBinary(domain::DO, max_arity, constmap, model_params::MP, obj2id::D, init::S, goal::G) where {DO,D,MP<:NamedTuple,S,G}
        @assert issubset((:message_passes, :residual), keys(model_params)) "Parameters of the model are not fully specified"
        @assert (init === nothing || goal === nothing) "Fixing init and goal state is bizzaare, as the extractor would always create a constant"
        new{DO,MP,D,S,G}(domain, max_arity, constmap, model_params, obj2id, init, goal)
    end
end

AtomBinaryNoGoal{DO,MP,D} = AtomBinary{DO,MP,D,Nothing,Nothing} where {DO,MP,D}
AtomBinaryStart{DO,MP,D,S} = AtomBinary{DO,MP,D,S,Nothing} where {DO,MP,D,S<:KnowledgeBase}
AtomBinaryGoal{DO,MP,D,S} = AtomBinary{DO,MP,D,Nothing,S} where {DO,MP,D,S<:KnowledgeBase}

function AtomBinary(domain; message_passes=2, residual=:linear, kwargs...)
    dictmap(x) = Dict(reverse.(enumerate(sort(x))))
    model_params = (; message_passes, residual)
    max_arity =  maximum(length(a.args) for a in values(domain.predicates))
    constmap = Dict{Symbol,Int}(dictmap([x.name for x in domain.constants]))
    AtomBinary(domain, max_arity, constmap, model_params, nothing, nothing, nothing)
end

isspecialized(ex::AtomBinary) = ex.obj2id !== nothing
hasgoal(ex::AtomBinary) = ex.init_state !== nothing || ex.goal_state !== nothing


function AtomBinary(domain, problem; embed_goal=true, kwargs...)
    ex = AtomBinary(domain; kwargs...)
    ex = specialize(ex, problem)
    embed_goal ? add_goalstate(ex, problem) : ex
end

function Base.show(io::IO, ex::AtomBinary)
    if !isspecialized(ex)
        print(io, "Unspecialized extractor for ", ex.domain.name)
    else
        g = hasgoal(ex) ? "with" : "without"
        print(io, "Specialized extractor ", g, " goal for ", ex.domain.name, )
    end
end

"""
specialize(ex::AtomBinary{<:Nothing,<:Nothing}, problem)

initializes extractor for a given `problem` by initializing mapping 
from objects to id of vertices. Goals are not changed added to the 
extractor.
"""
function specialize(ex::AtomBinary, problem)
    obj2id = Dict(v.name => (;id = i, set = BitSet(i)) for (i, v) in enumerate(problem.objects))
    for k in keys(ex.constmap)
        i = length(obj2id) + 1
        obj2id[k] = (;id = i, set = BitSet(i))
    end
    AtomBinary(ex.domain, ex.max_arity, ex.constmap, ex.model_params, obj2id, nothing, nothing)
end

function (ex::AtomBinary)(state::GenericState)
    prefix = (ex.goal_state !== nothing) ? :start : ((ex.init_state !== nothing) ? :goal : nothing)
    kb = encode_state(ex, state, prefix)
    addgoal(ex, kb)
end

function encode_state(ex::AtomBinary, state::GenericState, prefix=nothing)
    message_passes, residual = ex.model_params
    x = unary_predicates(ex, state)
    kb = KnowledgeBase((; x1=x))
    n = size(x, 2)
    sₓ = :x1
    edge_structure = encode_edges(ex, :x1, state, prefix)
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
nunary_predicates(ex::AtomBinary, state)

Create matrix with one column per atom and encode by one-hot-encoding the type of an atom
"""
function unary_predicates(ex::AtomBinary, state)
    # There will be a Matrix, where each
    error("do unary_predicates")

    # encode constants
    idim = length(ex.nunanary_predicates) + length(ex.objtype2id) + length(ex.constmap)
    x = zeros(Float32, idim, length(ex.obj2id))
    x
end

function pred2set(ex::AtomBinary, x::Julog.Term)
    isempty(x.args) && return(BitSet())
    mapreduce(k -> ex.obj2id[k.name].set, union, x.args)
end

"""
    pos_in_arg(ex::AtomBinary, k::Int, p::Compound)

    position of an argument in the predicate
"""
function pos_in_arg(ex::AtomBinary, k::Int, p::Compound)
    findfirst(a -> ex.obj2id[a.name].id == k, p.args)
end

"""
    type_of_edge(ex::AtomBinary, i::Int, sa, sb)

    Index of type of an edge. Type of an edge is defined by the index of the object in the predicates `sa` and `sb`.
"""
function type_of_edge(ex::AtomBinary, k::Int, sa, sb)
    i, j = pos_in_arg(ex, k, sa), pos_in_arg(ex, k, sb)
    ex.max_arity*(i-1) + j
end

function encode_edges(ex::AtomBinary, pname::Symbol, preds, kid::Symbol)
    preds = collect(PDDL.get_facts(state))
    set_preds = map(Base.Fix1(pred2set, ex), preds) 
    for i in 1:length(preds) # Can be replaced with Combinatorics.preds
        sa = set_preds[i]
        for j in 2:length(preds)
            sb = set_preds[j]
            is = intersect(sa,sb)
            if !isempty(is)
                for k in is
                   println("intersection edge: ",i," --> ",j, " of type ",type_of_edge(ex, k, preds[i], preds[j]))
                end
            end

            sd = symdiff(sa,sb)
            if !isempty(sd)
                subsets = [preds[k].name for k in 1:length(preds) if issubset(sd, set_preds[k])]
                if !isempty(subsets)
                    println("there are symdiff edges of type ",subsets)
                end
            end
        end
    end
end


function add_goalstate(ex::AtomBinary, problem, goal=goalstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    exg = encode_state(ex, goal, :goal)
    AtomBinary(ex.domain, ex.multiarg_predicates, ex.nunanary_predicates, ex.objtype2id, ex.constmap, ex.model_params, ex.obj2id, nothing, exg)
end

function add_initstate(ex::AtomBinary, problem, start=initstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    exg = encode_state(ex, start, :start)
    AtomBinary(ex.domain, ex.multiarg_predicates, ex.nunanary_predicates, ex.objtype2id, ex.constmap, ex.model_params, ex.obj2id, exg, nothing)
end

function addgoal(ex::AtomBinaryStart, kb::KnowledgeBase)
    error("implement me")
    return (stack_hypergraphs(ex.init_state, kb))
end

function addgoal(ex::AtomBinaryGoal, kb::KnowledgeBase)
    error("implement me")
    return (stack_hypergraphs(kb, ex.goal_state))
end

function addgoal(ex::AtomBinaryNoGoal, kb::KnowledgeBase)
    return (kb)
end
