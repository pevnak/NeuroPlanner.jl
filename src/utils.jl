
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


@inline function _map_tuple(f::F, xs::Tuple) where {F<:Union{Function, Base.Fix1, Base.Fix2}}
    isempty(xs) && return(xs)
    tuple(f(first(xs)), _map_tuple(f, Base.tail(xs))...)
end

@inline function _map_tuple(f::F, xs::NamedTuple{KS}) where {KS,F<:Union{Function, Base.Fix1, Base.Fix2}}
    isempty(xs) && return(xs)
    NamedTuple{KS}(tuple(f(first(xs)), _map_tuple(f, Base.tail(xs))...))
end


@inline function _map_tuple(f::F, i::Type{Val{I}},n::Type{Val{N}}) where {I,N,F<:Union{Function, Base.Fix1, Base.Fix2}}
    I > N && return(tuple())
    I == N && return(tuple(f(I)))
    tuple(f(I), _map_tuple(f, Val{I+1}, n)...)
end

@inline _map_tuple(f::F, n::Type{Val{N}}) where {N,F<:Union{Function, Base.Fix1, Base.Fix2}} = _map_tuple(f, Val{1}, n)
