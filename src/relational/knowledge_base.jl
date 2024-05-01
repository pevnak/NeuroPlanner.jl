import Base.*
import Base.reduce
import Base.==

"""
struct KnowledgeBase{KB<:NamedTuple}
    kb::KB
end

a thin wrapper around NamedTuple, such that we have a control over dispatch
"""
struct KnowledgeBase{KS,VS}
    kb::NamedTuple{KS,VS}
end

function Base.show(io::IO, kb::KnowledgeBase{KS,VS}) where {KS,VS}
    print(io, "KnowledgeBase: (", join(KS, ","), ")")
end

KnowledgeBase() = KnowledgeBase(NamedTuple())
function KnowledgeBase(xs::Vector{<:Pair{Symbol,Any}})
    KS = map(first, xs)
    VS = map(Base.Fix2(getindex,2), xs)
    KnowledgeBase(NamedTuple{KS}(VS))
end

Base.getindex(kb::KnowledgeBase, k::Symbol) = kb.kb[k]
Base.getindex(kb::KnowledgeBase, k::Integer) = kb.kb[k]
Base.keys(kb::KnowledgeBase) = keys(kb.kb)
Base.tail(kb::KnowledgeBase) = KnowledgeBase(Base.tail(kb.kb))
Base.isempty(kb::KnowledgeBase) = isempty(kb.kb)
Base.lastindex(kb::KnowledgeBase) = length(kb.kb)
append(kb::KnowledgeBase, k::Symbol, x) = KnowledgeBase(merge(kb.kb, NamedTuple{(k,)}((x,))))
function Base.replace(kb::KnowledgeBase, k::Symbol, v)
    l = Accessors.PropertyLens{k}() ∘ Accessors.PropertyLens{:kb}()
    set(kb, l, v)
end

"""
atoms(kb::KnowledgeBase)

keys of items in knowledgebase that evaluates to true
"""
function atoms(kb::KnowledgeBase)
    ks = filter(k -> kb[k] isa AbstractArray, keys(kb))
    KnowledgeBase(kb.kb[ks])
end

MLUtils.batch(xs::Vector{<:KnowledgeBase}) = _catobs_kbs(xs)

"""
struct KBEntry{T} <: AbstractMatrix{T}
    ii::Vector{Int}
end

represent matrix as indices `ii` to items stored in knowledgebase. I.e. The matrix is equal to 
kb[E][:,ii], where `kb` is some suitable storage, like NamedTuple. The advantage is 
that the data are stored lazily and instantiated just before use.

E --- is a name of the entry taken from the knowledge base
T --- is the datatype
N --- is the number of items in the knowledge base. When
"""
struct KBEntry{T} <: AbstractMatrix{T}
    e::Symbol
    ii::Vector{Int}
end

function KBEntry(T::DataType, e, ii)
    KBEntry{T}(Symbol(e),ii)
end

KBEntry(e, ii) = KBEntry(Float32, e, ii)


Base.show(io::IO, A::KBEntry{T}) where {T} = print(io, "KBEntry{$(A.e)} with $(length(A.ii)) items")
Base.show(io::IO, ::MIME"text/plain", A::KBEntry{T}) where {T} = print(io, "KBEntry{$(A.e)} with $(length(A.ii)) items")
Base.show(io::IO, ::MIME"text/plain", @nospecialize(n::ArrayNode{<:KBEntry})) = print(io, "(: × ", length(n.data.ii), ")")
Base.size(A::KBEntry{T}) where {T} = ((:), length(A.ii))
Base.size(A::KBEntry{T}, d) where {T} = (d == 1) ? (:) : length(A.ii)
Base.Matrix(kb::KnowledgeBase, x::KBEntry{T}) where {T} = kb[x.e][:, x.ii]
Base.Matrix(X::KBEntry) = error("cannot instantiate Matrix from KBEntry without a KnowledgeBase, use Matrix(kb::KnowledgeBase, A::KBEntry)")
Base.getindex(X::KBEntry, idcs) = _getindex(X, idcs)
Base.axes(X::KBEntry, d) = d == 1 ? (:) : (1:length(X.ii))
function ==(a::KBEntry, b::KBEntry)
    a.e != b.e && return (false)
    return (a.ii == b.ii)
end
_getindex(x::KBEntry{T}, i) where {T} = KBEntry{T}(x.e, x.ii[i])
_getindex(x::KBEntry{T}) where {T} = x
_getindex(x::KBEntry{T}, i::Integer) where {T} = KBEntry{T}(x.e, x.ii[i:i])
MLUtils.numobs(a::KBEntry) = length(a.ii)
HierarchicalUtils.NodeType(::Type{KBEntry}) = HierarchicalUtils.LeafNode()

ChainRulesCore.ProjectTo(x::KBEntry{T}) where {T} = ProjectTo{KBEntry}(; E, T, ii=x.ii)

########
#   We do not do reduce, as it is potentially very dangerous as it might be dissinchronized with the knowledge base
########
function Mill.catobs(A::KBEntry{T}, B::KBEntry{T}) where {T}
    error("The catobs is not implemented for \"KBEntry\", as it might not be safe without KnowledgeBase")
end

function reduce(::typeof(Mill.catobs), As::Tuple{Vararg{KBEntry{T}}}) where {T}
    error("The catobs is not implemented for \"KBEntry\", as it might not be safe without KnowledgeBase")
end

function reduce(::typeof(Mill.catobs), As::Vector{<:KBEntry{T}}) where {T}
    error("The catobs is not implemented for \"KBEntry\", as it might not be safe without KnowledgeBase")
end

########
#   THe dangerousness of getindex is questionably and will be removed on first trouble
########
function Base.getindex(X::KBEntry{T}, idcs...) where {T}
    D = length(X.ii)
    if first(idcs) isa Colon
        idcs = idcs[2:end]
    end
    if first(idcs) == Base.Slice(Base.OneTo(D))
        idcs = idcs[2:end]
    end
    length(idcs) > 1 && error("cannot subsample rows of embedding vectors")
    _getindex(X, idcs...)
end


#########
# concatenation of knowledge bases
#########

# Concatenation of knowledge base is difficult, since KBEntries are essentially
# views into the knowledgebase and therefore we need to ensure that when
# we concatenate two knowledge bases for batching purposes, we correctly 
# indexes into the original arrays. An example of a possible  problem 
# occurs when the knowledge base contains longer array then is the 
# maximum index in the KBEntry. In this case, problem might happen.
# The implementation would a very nice user-case for contextual dispatch, 
# but since Casette or IRTools would take ages to compile, we just manually 
# copy things by ourselves. On the end, this code is experimental.

reduce(::typeof(Mill.catobs), as::AbstractVector{<:KnowledgeBase}) = _catobs_kbs(as)

function _catobs_kbs(as::AbstractVector{<:KnowledgeBase{KS,VS}}) where {KS,VS}
    offsets = _compute_offsets(as)
    vs = map(KS) do k
        _catobs_kbs(offsets, [a[k] for a in as])
    end
    KnowledgeBase(NamedTuple{KS}(vs))
end

function _catobs_kbs(offsets, as::AbstractVector{<:AbstractMatrix})
    reduce(hcat, as)
end

function _catobs_kbs(offsets, as::AbstractVector{<:ArrayNode})
    ArrayNode(_catobs_kbs(offsets, [a.data for a in as]))
end

function _catobs_kbs(offsets, as::AbstractVector{<:KBEntry{T}}) where {T}
    E = first(as).e 
    all(E == a.e for a in as) || error("cannot concatenate KBEntries pointing to different keys in KnowledgeBase")
    i = [a.ii for a in as]
    ii = reduce(vcat, [a.ii .+ o for (a, o) in zip(as, offsets[E])])
    KBEntry{T}(E, ii)
end

function _catobs_kbs(offsets, as::AbstractVector{<:ProductNode{T,<:Nothing}}) where {T}
    as = [a.data for a in as]
    xs = T(_catobs_kbs(offsets, [a[i] for a in as]) for i in keys(as[1]))
    ProductNode(xs)
end

function _catobs_kbs(offsets, as::AbstractVector{<:BagNode})
    d = [a.data for a in as]
    bags = reduce(vcat, [n.bags for n in as])
    BagNode(_catobs_kbs(offsets, d), bags, nothing)
end

function _catobs_kbs(offset, as::AbstractVector{<:MaskedNode})
    xs = _catobs_kbs(offset, [a.data for a in as])
    MaskedNode(xs, reduce(vcat, [x.mask for x in as]))
end

function _compute_offsets(as::AbstractVector{<:KnowledgeBase{KS,VS}}) where {KS,VS}
    o = map(KS) do k
        o = 0
        c = zeros(Int, length(as))
        for (i, a) in enumerate(as)
            c[i] = o
            o += numobs(a[k])
        end
        return (c)
    end
    NamedTuple{KS}(o)
end

###########
#   Plumbing to Mill --- propagation with Knowledge base
###########
function (m::Mill.ArrayModel)(kb::KnowledgeBase, x::ArrayNode{<:KBEntry})
    xx = Matrix(kb, x.data)
    m.m(xx)
end

(m::BagModel)(kb::KnowledgeBase, x::BagNode{<:AbstractMillNode}) = m.bm(m.a(m.im(kb, x.data), x.bags))
(m::BagModel)(kb::KnowledgeBase, x::BagNode{Missing}) = m.bm(m.a(x.data, x.bags))

(m::MaskedModel)(kb::KnowledgeBase, x::MaskedNode{<:AbstractMillNode}) = m.m(kb, x.data) .* x.mask'


@generated function (m::ProductModel{<:NamedTuple{KM}})(kb::KnowledgeBase, x::ProductNode{<:NamedTuple{KD}}) where {KM,KD}
    @assert issubset(KM, KD)
    chs = map(KM) do k
        :(m.ms.$k(kb, x.data.$k))
    end
    quote
        m.m(LazyVCatMatrix($(chs...)))
    end
end

@generated function (m::ProductModel{T})(kb::KnowledgeBase, x::ProductNode{U}) where {T<:Tuple,U<:Tuple}
    l1 = T.parameters |> length
    l2 = U.parameters |> length
    @assert l1 ≤ l2 "Applied ProductModel{<:Tuple} has more children than ProductNode"
    chs = map(1:l1) do i
        :(m.ms[$i](kb, x.data[$i]))
    end
    quote
        m.m(LazyVCatMatrix($(chs...)))
    end
end


###########
#   Plumbing to Mill --- making reflect model nice
###########
import Mill: reflectinmodel, _reflectinmodel

function _reflectinmodel(kb::KnowledgeBase, x::AbstractBagNode, fm, fa, fsm, fsa, s, args...)
    c = Mill.stringify(s)
    im, d = _reflectinmodel(kb, x.data, fm, fa, fsm, fsa, s * Mill.encode(1, 1), args...)
    agg = haskey(fsa, c) ? fsa[c](d) : fa(d)
    d = size(BagModel(im, agg)(kb, x), 1)
    bm = haskey(fsm, c) ? fsm[c](d) : fm(d)
    m = BagModel(im, agg, bm)
    m, size(m(kb, x), 1)
end

function _reflectinmodel(kb::KnowledgeBase, x::AbstractProductNode, fm, fa, fsm, fsa, s, ski, args...)
    c = Mill.stringify(s)
    n = length(x.data)
    ks = keys(x.data)
    ms, ds = zip([_reflectinmodel(kb, x.data[k], fm, fa, fsm, fsa, s * Mill.encode(i, n), ski, args...)
                  for (i, k) in enumerate(ks)]...) |> collect
    ms = Mill._remap(x.data, ms)
    m = if haskey(fsm, c)
        fsm[c](sum(ds))
    elseif ski && n == 1
        identity
    else
        fm(sum(ds))
    end
    m = ProductModel(ms, m)
    m, size(m(kb, x), 1)
end

function _reflectinmodel(kb::KnowledgeBase, x::ArrayNode, fm, fa, fsm, fsa, s, ski, ssi, ai)
    xx = Matrix(kb, x.data)
    c = Mill.stringify(s)
    r = size(xx, 1)
    m = if haskey(fsm, c)
        fsm[c](r)
    elseif ssi && r == 1
        identity
    else
        fm(r)
    end |> ArrayModel
    m = Mill._make_imputing(xx, m, ai)
    m, size(m(kb, x), 1)
end


function _reflectinmodel(kb::KnowledgeBase, x::AbstractMaskedNode, fm, fa, fsm, fsa, s, ski, args...)
    m, s = _reflectinmodel(kb, x.data, fm, fa, fsm, fsa, s, ski, args...)
    MaskedModel(m), s
end


###########
#  replacement in mill structures.
###########

function Base.replace(x::KBEntry{T}, ps::Pair{Symbol,Symbol}...) where {T}
    for (a, b) in ps
        x.e == a && return (KBEntry{T}(b, x.ii))
    end
    return (x)
end

Base.replace(x::ArrayNode, ps::Pair{Symbol,Symbol}...) = ArrayNode(replace(x.data, ps...), x.metadata)
Base.replace(x::BagNode, ps::Pair{Symbol,Symbol}...) = BagNode(replace(x.data, ps...), x.bags, x.metadata)
Base.replace(x::ProductNode, ps::Pair{Symbol,Symbol}...) = ProductNode(map(k -> replace(k, ps...), x.data), x.metadata)
Base.replace(x::MaskedNode, ps::Pair{Symbol,Symbol}...) = MaskedNode(map(k -> replace(k, ps...), x.data), x.mask)
