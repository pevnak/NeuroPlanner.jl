using PDDL: get_facts, get_args
"""
```julia
struct AtomBinary{DO,EB,MP,D,S,G}
    domain::DO
    edgebuilder::EB
    max_arity::Int
    constmap::Dict{Symbol,Int}
    actionmap::Dict{Symbol,Int}
    model_params::MP
    obj2id::D
    init_state::S           # contains atoms of the init state
    goal_state::G           # contains atoms of the goal state
end
```

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
struct AtomBinary{DO,EB,MP,D,S,G}
    domain::DO
    edgebuilder::EB
    max_arity::Int
    constmap::Dict{Symbol,Int}
    actionmap::Dict{Symbol,Int}
    model_params::MP
    obj2id::D
    init_state::S           # contains atoms of the init state
    goal_state::G           # contains atoms of the goal state
    function AtomBinary(domain::DO, edgebuilder::EB, max_arity, constmap, actionmap, model_params::MP, obj2id::D, init::S, goal::G) where {DO,EB,D,MP<:NamedTuple,S,G}
        @assert issubset((:message_passes, :residual), keys(model_params)) "Parameters of the model are not fully specified"
        @assert (init === nothing || goal === nothing) "Fixing init and goal state is bizzaare, as the extractor would always create a constant"
        new{DO,EB,MP,D,S,G}(domain, edgebuilder, max_arity, constmap, actionmap, model_params, obj2id, init, goal)
    end
end

AtomBinaryNoGoal{DO,EB,MP,D} = AtomBinary{DO,EB,MP,D,Nothing,Nothing} where {DO,EB,MP,D}
AtomBinaryStart{DO,EB,MP,D,S} = AtomBinary{DO,EB,MP,D,S,Nothing} where {DO,EB,MP,D,S<:AbstractVector{<:Julog.Term}}
AtomBinaryGoal{DO,EB,MP,D,G} = AtomBinary{DO,EB,MP,D,Nothing,G} where {DO,EB,MP,D,G<:AbstractVector{<:Julog.Term}}

function AtomBinary(domain; message_passes=2, residual=:linear, edgebuilder = FeaturedEdgeBuilder, kwargs...)
    dictmap(x) = Dict(reverse.(enumerate(sort(x))))
    model_params = (; message_passes, residual)
    max_arity =  maximum(length(a.args) for a in values(domain.predicates))     # maximum arity of a predicate
    constmap = Dict{Symbol,Int}(dictmap([x.name for x in domain.constants]))    # identification of constants
    actionmap = Dict{Symbol,Int}(dictmap(collect(keys(domain.predicates))))     # codes of actions
    AtomBinary(domain, edgebuilder, max_arity, constmap, actionmap, model_params, nothing, nothing, nothing)
end


isspecialized(ex::AtomBinary) = ex.obj2id !== nothing
hasgoal(ex::AtomBinary) = ex.init_state !== nothing || ex.goal_state !== nothing

AtomBinaryFE(domain; kwargs...) = AtomBinary(domain; edgebuilder = FeaturedHyperEdgeBuilder, kwargs...)
AtomBinaryFENA(domain; kwargs...) = AtomBinary(domain; edgebuilder = FeaturedHyperEdgeBuilderNA, kwargs...)
AtomBinaryME(domain; kwargs...) = AtomBinary(domain; edgebuilder = MultiHyperEdgeBuilder, kwargs...)
AtomBinaryFE(domain, problem; kwargs...) = AtomBinary(domain, problem; edgebuilder = FeaturedHyperEdgeBuilder, kwargs...)
AtomBinaryFENA(domain, problem; kwargs...) = AtomBinary(domain, problem; edgebuilder = FeaturedHyperEdgeBuilderNA, kwargs...)
AtomBinaryME(domain, problem; kwargs...) = AtomBinary(domain, problem; edgebuilder = MultiHyperEdgeBuilder, kwargs...)

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
    obj2id = Dict(v.name => i for (i, v) in enumerate(problem.objects))
    for k in keys(ex.constmap)
        obj2id[k] = length(obj2id) + 1
    end
    AtomBinary(ex.domain, ex.edgebuilder, ex.max_arity, ex.constmap, ex.actionmap, ex.model_params, obj2id, nothing, nothing)
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

"""
    group_facts(ex::AtomBinary, facts::Vector{Julog.Term})

    Create a list, where each object ID has a list of 
    facts and position of the objects inside facts
"""
function group_facts(ex::AtomBinary, facts::Vector{Julog.Term})

    # init info about predicates    
    predicates = tuple(keys(ex.domain.predicates)...)
    arrities = map(k -> length(ex.domain.predicates[k].args), predicates)
    parr = map(f -> arrities[_inlined_search(f.name, predicates)], facts)
    
    # init structure for storing information about facts
    NT = @NamedTuple{position::Int64, atom_id::Int64}
    ids_in_facts = [NT[] for _ in 1:length(ex.obj2id)]

    for arrity in 1:maximum(arrities)
        add_object_to_group!(ids_in_facts, facts, ex.obj2id, parr, Val(arrity))
    end
    ids_in_facts
end
   
@generated function add_object_to_group!(ids_in_facts, facts, obj2id, parr,  arrity::Val{N}) where {N}
    stmts = quote
    end
    for j in 1:N
        push!(stmts.args, :(o = a.args[$(j)]))
        push!(stmts.args, :(oid = obj2id[o.name]))
        push!(stmts.args, :(push!(ids_in_facts[oid], (;position = $(j), atom_id = i))))
    end
    quote 
        for i in 1:length(facts)
            parr[i] != $(N) && continue
            a = facts[i]
            $(stmts)
        end
    end
end

function encode_edges(ex::AtomBinary, atoms::Vector{Julog.Term}, kid::Symbol)
    ids_in_atoms = group_facts(ex, atoms)
    l = length(atoms)
    capacity = ex.max_arity * l * (l + 1) ÷ 2
    eb = ex.edgebuilder(2, capacity, l, ex.max_arity^2)
    encode_edges(ex, eb, ids_in_atoms, kid)
end

function encode_edges(ex, eb, ids_in_atoms, kid)
    for as in ids_in_atoms
        for i in 1:length(as)
            for j in 2:length(as)
               k = _type_of_edge(ex, as[i].position, as[j].position)
               push!(eb, (as[i].atom_id, as[j].atom_id), k)
            end
        end
    end
    construct(eb, kid)
end

@inline function _type_of_edge(ex::AtomBinary, i, j)
    ex.max_arity*(i-1) + j
end

function add_goalstate(ex::AtomBinary, problem, goal=goalstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    @set ex.goal_state = collect(get_facts(goal))
end

function add_initstate(ex::AtomBinary, problem, start=initstate(ex.domain, problem))
    ex = isspecialized(ex) ? ex : specialize(ex, problem)
    @set ex.init_state = collect(get_facts(start))
end

"""
    atoms, gid = add_goalatoms(ex, atoms)

    Add atoms determining goal (or init) state to atoms and return indices of goal state.
    The addition is made such that goal states are always first. 
"""
function add_goalatoms(ex::AtomBinaryStart, atoms)
    gid = 1:length(atoms)
    vcat(atoms, ex.init_state), gid
end

function add_goalatoms(ex::AtomBinaryGoal, atoms)
    gid = 1:length(ex.goal_state)
    vcat(ex.goal_state, atoms), gid
end

function add_goalatoms(ex::AtomBinaryNoGoal, atoms)
    gid = 0:-1
    atoms, gid
end

