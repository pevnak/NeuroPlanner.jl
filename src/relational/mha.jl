
abstract type AbstractAttention end

struct MultiheadAttention{Q<:Dense, K<:Dense, V<:Dense, O<:Dense, DP<:Dropout} <: AbstractAttention
    head::Int
    future::Bool
    iqproj::Q
    ikproj::K
    ivproj::V
    oproj::O
    drop::DP
end

# Flux.functor(mh::MultiheadAttention) = (mh.iqproj, mh.ikproj, mh.ivproj, mh.oproj, mh.drop), m -> MultiheadAttention(mh.head, mh.future, m...)

"""
    MultiheadAttention(head::Int, is::Int, hs::Int, os::Int;
                       future::Bool=true, pdrop = 0.1)
Multihead dot product Attention Layer, `head` is the number of head,
`is` is the input size, `hs` is the hidden size of input projection layer of each head,
`os` is the output size. When `future` is `false`, the k-th token can't see tokens at > k.
`pdrop` is the dropout rate.
"""
function MultiheadAttention(head::Int, is::Int, hs::Int, os::Int; future::Bool=true, pdrop = 0.1) 
    MultiheadAttention(head, future, Dense(is, hs*head), Dense(is, hs*head), Dense(is, hs*head), Dense(hs*head, os), Dropout(pdrop))
end                                                                        


function Base.show(io::IO, mh::MultiheadAttention)
    hs = div(size(mh.iqproj.weight)[1], mh.head)
    is = size(mh.iqproj.weight)[end]
    os = size(mh.oproj.weight)[1]

    print(io, "MultiheadAttention(")
    print(io, "head=$(mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "$(is)=>$(os)")

    if Flux.istraining()
        print(io, ", dropout=$(mh.drop.p))")
    else
        print(io, ")")
    end
end


function (mha::MultiheadAttention)(x::AbstractArray{<:Any,3}, mask=nothing, p = nothing)
    q = mha.iqproj(x)
    k = mha.ikproj(x)
    v = mha.ivproj(x)
    a = NeuralAttentionlib.multihead_qkv_attention(4, q, k, v, mask, p)
    mha.oproj(a)
end

(mha::MultiheadAttention)(x::AbstractMatrix, mask=nothing, p = nothing) = reshape(mha(reshape(x, size(x)...,1)), :, size(x,2))
