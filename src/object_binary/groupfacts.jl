"""
    struct PredicateInfo{N,I}
        predicates::NTuple{N,Symbol}
        arrities::NTuple{N,I}
        id2fid::NTuple{N,I}
    end


    Preprocess informations about the predicates, which are useful for the conversion of 
    the PDDL.GenericState to IntState and then to the graph.

    predicates --- is a tuple containing names of predicates
    arrities --- is a tuple containing arities if predicates
    id2fid --- is a tuple containing optional map of predicate ids to feature ids. This is useful,
    when nullary and unary predicates are features of vertices and predicates of higher arities,
    are coded as multi-edges or featured-edges. The default is (1,2,3,...,N), which does not change
    the map.
    nunary --- number of predicates with zero or one arity
    nary --- number of predicates with arity two or more
"""
struct PredicateInfo{N,I}
    predicates::NTuple{N,Symbol}
    arrities::NTuple{N,I}
    id2fid::NTuple{N,I}
    nunary::I
    nary::I
end

function PredicateInfo(domain::GenericDomain)
    predicates = tuple(keys(domain.predicates)...)
    arrities = map(k -> length(domain.predicates[k].args), predicates)
    id2fid = tuple(collect(1:length(predicates))...)
    nunary = sum(arrities .≤ 1)
    nary = sum(arrities .> 1)
    PredicateInfo(predicates, arrities, id2fid, nunary, nary)
end

"""
    struct IntState{N,I<:Integer}
        name::I
        args::NTuple{N,I}
    end

    IntState is an attempt to convert states to Integers,
    where `name` is the index of the predicate and `args` are ids of matched objects

"""
struct IntState{N,I<:Integer}
    name::I
    args::NTuple{N,I}
end

# Base.show(io::IO, s::IntState) = print(io, "IntState(", s.name," (",join(s.args", "),"))")


"""
    intstates(domain::GenericDomain, obj2id::Dict{Symbol,<:Integer}, facts::Vector{<:Term})
    intstates(domain::GenericDomain, obj2id::Dict{Symbol,<:Integer}, predicates::NTuple{N,Symbol}, arrities::NTuple{N, <:Integer}, facts::Vector{<:Term}) where {N}    

    convert facts stored in PDDL.Terms to a tuple of IntStates, where each
"""
function intstates(domain::GenericDomain, obj2id::Dict{Symbol,<:Integer}, facts::Vector{<:Term})
    pifo = PredicateInfo(domain)
    intstates(domain, obj2id, pifo, facts)
end

function intstates(domain::GenericDomain, obj2id::Dict{Symbol,<:Integer}, pifo::PredicateInfo, facts::Vector{<:Term})
    arrities = pifo.arrities
    predicates = pifo.predicates
    pids = map(f -> _inlined_search(f.name, predicates), facts)
    map(0:maximum(arrities)) do arrity
        mask = [arrities[pid] == arrity for pid in pids]
        intstates(obj2id, pifo, facts, pids, mask, Val(arrity))
    end
end

function intstates(obj2id::Dict{Symbol,I}, pifo::PredicateInfo, facts::Vector{<:Term}, pids,  mask, arity::Val{N}) where {N,I<:Integer}
    id2fid = pifo.id2fid
    o = Vector{IntState{N,I}}(undef, sum(mask))
    index = 1
    for i in 1:length(mask)
        mask[i] || continue
        f = facts[i]
        # o[index] = IntState(id2fid[pids[i]], _map_tuple(j -> obj2id[f.args[j].name], arity))
        o[index] = IntState(id2fid[pids[i]], args2ids(obj2id, f, arity))
        index += 1
    end
    o
end

@generated function args2ids(obj2id, f, arrity::Val{N}) where {N}
    chs = map(1:N) do j
        :(obj2id[f.args[$(j)].name])
    end
    quote
        tuple($(chs...))
    end
end


function merge_states(s1, s2)
    map(vcat, s1, s2)
end

"""
    group_facts(domain, obj2id::Dict{Symbol,<:Integer}, predicates::NTuple{N,Symbol}, facts::Vector{<:Term})
    group_facts(domain::GenericDomain, obj2id::Dict{Symbol,<:Integer}, nullary, unary, nary, facts::Vector{<:Term})

    Create a structure, where atoms are divided according to predicate names and their arguments are converted
    to integers using obj2id. The structure separates nullary, unary, and n-ary predicates, because they are handled 
    differently.


    domain --- domain file generic
    obj2id --- map from names to ids
    predicates --- predicates to be added. They will be internally separated to nullary, unary, or N-arry using information in domain
"""
function group_facts(domain::GenericDomain, obj2id::Dict{Symbol,<:Integer}, facts::Vector{<:Term})
    nullary = tuple(filter(k -> isempty(domain.predicates[k].args), keys(domain.predicates))...)
    unary = tuple(filter(k -> length(domain.predicates[k].args) == 1, keys(domain.predicates))...)
    binary = tuple(filter(k -> length(domain.predicates[k].args) == 2, keys(domain.predicates))...)
    ternary = tuple(filter(k -> length(domain.predicates[k].args) == 3, keys(domain.predicates))...)
    group_facts(domain, obj2id, nullary, unary, binary, ternary, facts)
end

function group_facts(domain::GenericDomain, obj2id::Dict{Symbol,<:Integer}, nullary, unary, binary, ternary, facts::Vector{<:Term})
    # we first mark, where the predicates occur
    nullary_occurences = falses(length(nullary))
    unary_occurences = falses(length(facts), length(unary))
    binary_occurences = falses(length(facts), length(binary))
    ternary_occurences = falses(length(facts), length(ternary))
    for (i, f) in enumerate(facts)
        col = _inlined_search(f.name, nullary)
        (col > 0) && (nullary_occurences[col] = true)
        col = _inlined_search(f.name, unary)
        (col > 0) && (unary_occurences[i, col] = true)
        col = _inlined_search(f.name, binary)
        (col > 0) && (binary_occurences[i, col] = true)
        col = _inlined_search(f.name, ternary)
        (col > 0) && (ternary_occurences[i, col] = true)
    end

    # then we construct compact representation
    (;  ternary = _mapenumerate_tuple(ternary) do col, k
            k => factargs2id(obj2id, facts, (@view ternary_occurences[:, col]), Val(3))
        end,
        binary = _mapenumerate_tuple(binary) do col, k
            k => factargs2id(obj2id, facts, (@view binary_occurences[:, col]), Val(2))
        end,
        unary = _mapenumerate_tuple(unary) do col, k
            k => map(first, factargs2id(obj2id, facts, (@view unary_occurences[:, col]), Val(1)))
        end,
        nullary = nullary_occurences,
    );
end

function factargs2id(obj2id::Dict{Symbol,<:Integer}, facts, mask, arity::Val{N}) where {N}
    o = Vector{NTuple{N,Int}}(undef, sum(mask))
    index = 1
    for i in 1:length(mask)
        mask[i] || continue
        p = facts[i]
        o[index] = _map_tuple(j -> obj2id[p.args[j].name], arity)
        index += 1
    end
    o
end
