import Base.*
import Base.reduce

"""
struct KnowledgeBase{KB<:NamedTuple}
	kb::KB
end

a thin wrapper around NamedTuple, such that we have a control over dispatch
"""
struct KnowledgeBase{KS, VS}
	kb::NamedTuple{KS, VS}
end

function Base.show(io::IO, kb::KnowledgeBase{KS, VS}) where {KS, VS}
	print(io, "KnowledgeBase: (",join(KS, ","),")");
end

Base.getindex(kb::KnowledgeBase, k::Symbol) = kb.kb[k]
Base.keys(kb::KnowledgeBase) = keys(kb.kb)
append(kb::KnowledgeBase, k::Symbol, x) = KnowledgeBase(merge(kb.kb,NamedTuple{(k,)}((x,))))
function atoms(kb::KnowledgeBase)
    ks = filter(k -> kb[k] isa AbstractArray, keys(kb))
    KnowledgeBase(kb.kb[ks])
end

"""
struct KBEntry{E,T} <: AbstractMatrix{T}
    ii::Vector{Int}
end

represent matrix as indices `ii` to items stored in knowledgebase. I.e. The matrix is equal to 
kb[E][:,ii], where `kb` is some suitable storage, like NamedTuple. The advantage is 
that the data are stored lazily and instantiated just before use.

E --- is a name of the entry taken from the knowledge base
T --- is the datatype
N --- is the number of items in the knowledge base. When
"""
struct KBEntry{E,T} <: AbstractMatrix{T}
    ii::Vector{Int}
end

function KBEntry(T::DataType, e, ii)
	KBEntry{Symbol(e), T}(ii)
end

KBEntry(e, ii) = KBEntry(Float32, e, ii)


Base.show(io::IO, A::KBEntry{E,T}) where {E,T} = print(io, "KBEntry{$(E)} with $(length(A.ii)) items")
Base.show(io::IO, ::MIME"text/plain", A::KBEntry{E,T}) where {E,T} = print(io, "KBEntry{$(E)} with $(length(A.ii)) items")
Base.show(io::IO, ::MIME"text/plain", @nospecialize(n::ArrayNode{<:KBEntry})) = print(io, "(: × ", length(n.data.ii),")")
Base.size(A::KBEntry{E,T}) where {E,T} = ((:), length(A.ii))
Base.size(A::KBEntry{E,T}, d) where {E,T} = (d == 1) ? (:) : length(A.ii)
Base.Matrix(kb::KnowledgeBase, X::KBEntry{E,T}) where {E,T}  = kb[E][:,X.ii]
Base.Matrix(X::KBEntry) = error("cannot instantiate Matrix from KBEntry without a KnowledgeBase, use Matrix(kb::KnowledgeBase, A::KBEntry)")
Base.getindex(X::KBEntry, idcs)  = _getindex(X, idcs)
Base.axes(X::KBEntry, d) = d == 1 ? (:) : (1:length(X.ii))
_getindex(x::KBEntry{E,T}, i) where {E,T} = KBEntry{E,T}(x.ii[i])
_getindex(x::KBEntry{E,T}, i::Integer) where {E,T} = KBEntry{E,T}(x.ii[i:i])
Base.hcat(as::T...) where {T <: KBEntry} = reduce(catobs, as)
Mill.nobs(a::KBEntry) = length(a.ii)
Mill.catobs(as::KBEntry) = reduce(catobs, as)
HierarchicalUtils.NodeType(::Type{KBEntry}) = HierarchicalUtils.LeafNode()
function Mill.catobs(A::KBEntry{E,T}, B::KBEntry{E,T}) where {E,T}
	KBEntry{E,T}(A.kb, vcat(A.ii, B.ii))
end

function reduce(::typeof(Mill.catobs), As::Tuple{Vararg{KBEntry{E,T}}}) where {E,T}
	KBEntry{E,T}(reduce(vcat, [a.ii for a in As]))
end

function reduce(::typeof(Mill.catobs), As::Vector{<:KBEntry{E,T}}) where {E,T}
	KBEntry{E,T}(reduce(vcat, [a.ii for a in As]))
end

function Base.getindex(X::KBEntry{E,T}, idcs...) where {E,T}
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

# function ChainRulesCore.rrule(::typeof(Base.Matrix), kb::KnowledgeBase, b::KBEntry)
#     y = Matrix(kb, b)
#     println("custom rrule")
#     function foo_mul_pullback(ȳ)
#     	@show (typeof(ȳ), size(ȳ))
#         return(NoTangent(), NoTangent(), NoTangent())
#     end
#     return y, foo_mul_pullback
# end


###########
#	Let's plumb it to Mill
###########
function (m::Mill.ArrayModel)(kb::KnowledgeBase, x::ArrayNode{<:KBEntry})
	xx = Matrix(kb, x.data)
	m.m(xx)
end

(m::BagModel)(kb::KnowledgeBase, x::BagNode{<:AbstractMillNode}) = m.bm(m.a(m.im(kb, x.data), x.bags))
(m::BagModel)(kb::KnowledgeBase, x::BagNode{Missing}) = m.bm(m.a(x.data, x.bags))
@generated function (m::ProductModel{<:NamedTuple{KM}})(kb::KnowledgeBase, x::ProductNode{<:NamedTuple{KD}}) where {KM,KD}
    @assert issubset(KM, KD)
    chs = map(KM) do k
        :(m.ms.$k(kb, x.data.$k))
    end
    quote
        m.m(vcat($(chs...)))
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
        m.m(vcat($(chs...)))
    end
end


###########
#	Let's plumb it to Mill
###########
import Mill: reflectinmodel, _reflectinmodel

function _reflectinmodel(kb::KnowledgeBase,x::AbstractBagNode, fm, fa, fsm, fsa, s, args...)
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
