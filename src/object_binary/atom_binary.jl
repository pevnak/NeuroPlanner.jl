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
    actionmap::Dict{Symbol,Int}
    model_params::MP
    obj2id::D
    init_atoms::S
    goal_atoms::G
    function AtomBinary(domain::DO, max_arity, constmap, actionmap, model_params::MP, obj2id::D, init::S, goal::G) where {DO,D,MP<:NamedTuple,S,G}
        @assert issubset((:message_passes, :residual), keys(model_params)) "Parameters of the model are not fully specified"
        @assert (init === nothing || goal === nothing) "Fixing init and goal state is bizzaare, as the extractor would always create a constant"
        new{DO,MP,D,S,G}(domain, max_arity, constmap, actionmap, model_params, obj2id, init, goal)
    end
end

AtomBinaryNoGoal{DO,MP,D} = AtomBinary{DO,MP,D,Nothing,Nothing} where {DO,MP,D}
AtomBinaryStart{DO,MP,D,S} = AtomBinary{DO,MP,D,S,Nothing} where {DO,MP,D,S<:AbstractVector{<:Julog.Term}}
AtomBinaryGoal{DO,MP,D,S} = AtomBinary{DO,MP,D,Nothing,S} where {DO,MP,D,S<:AbstractVector{<:Julog.Term}}

function AtomBinary(domain; message_passes=2, residual=:linear, kwargs...)
    dictmap(x) = Dict(reverse.(enumerate(sort(x))))
    model_params = (; message_passes, residual)
    max_arity =  maximum(length(a.args) for a in values(domain.predicates))     # maximum arity of a predicate
    constmap = Dict{Symbol,Int}(dictmap([x.name for x in domain.constants]))    # identification of constants
    actionmap = Dict{Symbol,Int}(dictmap(collect(keys(domain.predicates))))     # codes of actions
    AtomBinary(domain, max_arity, constmap, actionmap, model_params, nothing, nothing, nothing)
end

isspecialized(ex::AtomBinary) = ex.obj2id !== nothing
hasgoal(ex::AtomBinary) = ex.init_atoms !== nothing || ex.goal_atoms !== nothing


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
    # T = BitSet
    N = ceil(Int, (length(problem.objects) + length(ex.constmap))/8)
    T = SBitSet{N, UInt8}

    obj2id = Dict(v.name => (;id = i, set = T(i)) for (i, v) in enumerate(problem.objects))
    for k in keys(ex.constmap)
        i = length(obj2id) + 1
        obj2id[k] = (;id = i, set = T(i))
    end
    AtomBinary(ex.domain, ex.max_arity, ex.constmap, ex.actionmap, ex.model_params, obj2id, nothing, nothing)
end

function (ex::AtomBinary)(state::GenericState)
    encode_state(ex, state)
end

function encode_state(ex::AtomBinary, state::GenericState, prefix=nothing)
    message_passes, residual = ex.model_params
    atoms = collect(get_facts(state))
    atoms, gii = add_goalatoms(ex, atoms)
    x = unary_predicates(ex, atoms, gii)
    kb = KnowledgeBase((; x1=x))
    n = size(x, 2)
    sₓ = :x1
    edge_structure = encode_edges(ex, atoms, :x1)
    for i in 1:message_passes
        input_to_gnn = last(keys(kb))
        ds = KBEntryRenamer(:x1, input_to_gnn)(edge_structure)
        kb = append(kb, layer_name(kb, "gnn"), ds)
        if residual !== :none #if there is a residual connection, add it 
            kb = add_residual_layer(kb, keys(kb)[end-1:end], n)
        end
    end
    s = last(keys(kb))
    kb = append(kb, :o, BagNode(ArrayNode(KBEntry(s, 1:n)), [1:n]))
end


"""
    nunary_predicates(ex::AtomBinary, atoms, gii)

    Create matrix with one column per atom and encode by one-hot-encoding the type (name) 
    of an atom. `gii` contains indices of the goal state, which are identified by `1` in the 
    last row.
"""
function unary_predicates(ex::AtomBinary, atoms, gii)
    # encode constants
    idim = length(ex.actionmap) + 1
    x = zeros(Float32, idim, length(atoms))
    for (i, p) in enumerate(atoms)
        x[ex.actionmap[p.name],i] = 1
    end
    x[end, gii] .= 1
    x
end


function encode_edges(ex::AtomBinary, atoms::Vector{Julog.Term}, kid::Symbol)
    set_atoms = map(Base.Fix1(parse_atoms, ex), atoms) 
    id2atoms = _id2atoms(ex, set_atoms)
    l = length(set_atoms)
    capacity = l * (l + 1) ÷ 2
    ebs = tuple([EdgeBuilderComp(2, capacity, l) for _ in 1:ex.max_arity^2]...)
    sbs = tuple([EdgeBuilderComp(2, ex.max_arity*capacity, l) for _ in 1:length(ex.actionmap)]...)
    encode_edges(ebs, sbs, ex, kid, set_atoms, id2atoms)
end

function encode_edges(ebs, sbs, ex::AtomBinary, kid::Symbol, set_atoms, id2atoms)
    l = length(set_atoms)
    inserted = falses(length(sbs))
    @inbounds for i in 1:l # Can be replaced with Combinatorics.atoms
        sa = set_atoms[i]
        for j in 2:l
            sb = set_atoms[j]

            # encoding intersection
            is = intersect(sa.set, sb.set)
            if !isempty(is)
                for k in is
                   ti = _type_of_edge(ex, k, sa, sb)
                   push!(ebs[ti], (i,j))
                end
            end

            # encoding symdiff
            sd = symdiff(sa.set,sb.set)
            if !isempty(sd)
                subsets2 = getindex_reduce(id2atoms, sd)
                if !isempty(subsets2)
                    fill!(inserted, false)
                    for k in subsets2
                        id = ex.actionmap[set_atoms[k].name] 
                        inserted[id] && continue
                        push!(sbs[id], (i,j))
                        inserted[id] = true
                    end
                end
            end
        end
    end
    ProductNode(map(Base.Fix2(construct, kid), tuple(ebs..., sbs...)))
end


"""
    parse_atom(ex::AtomBinary, atom::Julog.Term)

    convert arguments of an atom to ids `ex.obj2id`
    and return them as a tuple of maximum size of predicates 
    (filled by 0) and their bitset.

"""
function parse_atoms(ex::AtomBinary, atom::Julog.Term)
    T = typeof(first(values(ex.obj2id)).set)
    ids = zeros(Int, ex.max_arity)
    set = T()
    name = atom.name
    for (i, k) in enumerate(atom.args)
        e = ex.obj2id[k.name]
        ids[i] = e.id
        set = union(set, e.set)
    end
    return((;name, set, ids = tuple(ids...)))
end

"""
    _type_of_edge(ex::AtomBinary, i::Int, sa, sb)

    Index of type of an edge. Type of an edge is defined by the index of the object in the predicates `sa` and `sb`.
"""
function _type_of_edge(ex::AtomBinary, k::Int, sa, sb)
    i, j = _inlined_search(k, sa.ids), _inlined_search(k, sb.ids)
    ex.max_arity*(i-1) + j
end

"""

    _id2atoms(ex::AtomBinary, set_atoms)

    Create a map of objectids to atom_ids, such that we can quickly check
    in which atoms the object id occurs in
"""
function _id2atoms(ex::AtomBinary, set_atoms)
    N = ceil(Int, (length(set_atoms))/8)
    T = SBitSet{N,UInt8}    
    _id2atoms(T, ex, set_atoms)
end

function _id2atoms(T, ex::AtomBinary, set_atoms)
    id2atoms = [T() for _ in 1:length(ex.obj2id)]
    for (i,sa) in enumerate(set_atoms)
        for id in sa.ids 
            id == 0 && continue
            id2atoms[id] = union(id2atoms[id], T(i))
        end 
    end
    id2atoms
end


"""
    getindex_reduce(id2atoms, sd)

    compute `reduce(intersect, [id2atoms[k] for k in sd])` while knowing
    that there is at least one element and avoding to allocate the 
    intermediate array
"""
function getindex_reduce(id2atoms, sd)
    i, state = iterate(sd)
    v = id2atoms[i]
    next = iterate(sd, state)
    while(next !== nothing)
        i, state = next
        @inbounds v = intersect(v, id2atoms[i])
        next = iterate(sd, state)
    end
    return(v)
end




function add_goalstate(ex::AtomBinary, problem, goal=goalstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    @set ex.goal_atoms = collect(get_facts(goal))
end

function add_initstate(ex::AtomBinary, problem, start=initstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    @set ex.init_atoms = collect(get_facts(start))
end

"""
    atoms, gid = add_goalatoms(ex, atoms)

    Add atoms determining goal (or init) state to atoms and return indices of goal state.
    The addition is made such that goal states are always first. 
"""
function add_goalatoms(ex::AtomBinaryStart, atoms)
    gid = 1:length(atoms)
    vcat(atoms, ex.init_atoms), gid
end

function add_goalatoms(ex::AtomBinaryGoal, atoms)
    gid = 1:length(ex.goal_atoms)
    vcat(ex.goal_atoms, atoms), gid
end

