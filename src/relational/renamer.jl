"""
    KBEntryRenamer(from, to)

    traverse HMIL sample and rename KBEntry{from} to KBEntry{to}

```julia

julia> using NeuroPlanner: KBEntryRenamer, KBEntry 
julia> KBEntryRenamer(:x1,:gnn2)(KBEntry(:x1, [1,2,3]))
KBEntry{gnn2} with 3 items

```

"""

struct KBEntryRenamer
    from::Symbol
    to::Symbol
end

KBEntryRenamer(from_to::Pair{Symbol,Symbol}) = KBEntryRenamer(from_to[1],from_to[2])

function (kb::KBEntryRenamer)(x::KBEntry{T}) where {T}
    kb.from != x.e && return(x)
    KBEntry{T}(kb.to, x.ii)
end

function Base.replace(kb::KnowledgeBase, k::Symbol, v::Symbol)
    KBEntryRenamer(k,v)(kb)
end

(kb::KBEntryRenamer)(ds::BagNode) = BagNode(kb(ds.data), ds.bags, ds.metadata)
(kb::KBEntryRenamer)(ds::ArrayNode) = ArrayNode(kb(ds.data), ds.metadata)
@generated function (kb::KBEntryRenamer)(x::ProductNode{<:NamedTuple{KM}}) where {KM}
    chs = map(KM) do k
        :(kb(x.data.$k))
    end
    quote
        ProductNode(NamedTuple{$(KM)}(tuple($(chs...))))
    end
end

@generated function (kb::KBEntryRenamer)(x::ProductNode{U}) where {U<:Tuple}
    l1 = U.parameters |> length
    chs = map(1:l1) do i
        :(kb(x.data[$i]))
    end
    quote
        ProductNode(tuple($(chs...)))
    end
end
