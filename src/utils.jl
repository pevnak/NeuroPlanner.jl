
"""
    _mapenumerate_tuple(f::Function, xs::Tuple)

    output `map(f, enumerate(xs))`, but preserves the output to be `tuple`
"""
function _mapenumerate_tuple(f::Function, xs::Tuple)
    isempty(xs) && return(xs)
    tuple(f(1, first(xs)), _mapenumerate_tuple(f, 2, Base.tail(xs))...)
end

function _mapenumerate_tuple(f::Function, i::Int,  xs::Tuple)
    isempty(xs) && return(xs)
    tuple(f(i, first(xs)), _mapenumerate_tuple(f, i+1, Base.tail(xs))...)
end

"""
    _inlined_search(s::Symbol, i::Int,  xs::Tuple)
    _inlined_search(s::Symbol, xs::Tuple)

    find a first occurence of Symbol `s` in xs starting at `i`-th element of xs
    When symbol is not found, -1 is returned

```julia
julia> _inlined_search(:a, (:b,:a,:c))
2

julia> _inlined_search(:e, (:b,:a,:c))
-1
   
"""
@inline function _inlined_search(s, i::Int,  xs::Tuple)
    isempty(xs) && return(-1)
    s == first(xs) && return(i)
    return(_inlined_search(s, i+1, Base.tail(xs)))
end

@inline _inlined_search(s, xs::Tuple) = _inlined_search(s, 1, xs)


@inline function _map_tuple(f::F, xs::Tuple) where {F<:Union{Function, Base.Fix1, Base.Fix2}}
    isempty(xs) && return(xs)
    tuple(f(first(xs)), _map_tuple(f, Base.tail(xs))...)
end

@inline function _map_tuple(f::F, xs::NamedTuple{KS}) where {KS,F<:Union{Function, Base.Fix1, Base.Fix2}}
    isempty(xs) && return(xs)
    NamedTuple{KS}(tuple(f(first(xs)), _map_tuple(f, Base.tail(xs))...))
end


@inline function _map_tuple(f::F, i::Val{I},n::Val{N}) where {I,N,F<:Union{Function, Base.Fix1, Base.Fix2}}
    I > N && return(tuple())
    I == N && return(tuple(f(I)))
    tuple(f(I), _map_tuple(f, Val(I+1), n)...)
end

@inline _map_tuple(f::F, n::Val{N}) where {N,F<:Union{Function, Base.Fix1, Base.Fix2}} = _map_tuple(f, Val(1), n)
