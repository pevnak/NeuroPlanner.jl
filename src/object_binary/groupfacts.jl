"""
    group_facts(domain, obj2id::Dict{Symbol,<:Integer}, predicates::NTuple{N,Symbol}, facts::Vector{<:Term})

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
    nary = tuple(filter(k -> length(domain.predicates[k].args) > 1, keys(domain.predicates))...)

    # we first mark, where the predicates occur
    nullary_occurences = falses(length(nullary))
    unary_occurences = falses(length(facts), length(unary))
    nary_occurences = falses(length(facts), length(nary))
    for (i, f) in enumerate(facts)
        col = _inlined_search(f.name, nullary)
        (col > 0) && (nullary_occurences[col] = true)
        col = _inlined_search(f.name, unary)
        (col > 0) && (unary_occurences[i, col] = true)
        col = _inlined_search(f.name, nary)
        (col > 0) && (nary_occurences[i, col] = true)
    end

    # then we construct compact representation
    (;  nary = _mapenumerate_tuple(nary) do col, k
            N = length(domain.predicates[k].args)
            k => factargs2id(obj2id, facts, (@view nary_occurences[:, col]), Val(N))
        end,
        unary = _mapenumerate_tuple(unary) do col, k
            N = length(domain.predicates[k].args)
            k => factargs2id(obj2id, facts, (@view unary_occurences[:, col]), Val(N))
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
