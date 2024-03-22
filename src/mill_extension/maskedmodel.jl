"""
    MaskModel{T} <: AbstractMillModel

A model node for processing [`MaskedModel`](@ref)s. It applies a (sub)model `m` stored in it to data in 
an [`MaskedModel`](@ref).

# Examples
```jldoctest masked_model
julia> Random.seed!(0);
```

```jldoctest masked_model
julia> n = ProductNode(a=ArrayNode([0 1; 2 3]), b=ArrayNode([4 5; 6 7]))
ProductNode  # 2 obs, 16 bytes
  ├── a: ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes
  ╰── b: ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes

julia> mn = MaskedNode(n, BitVector([1, 0]))
MaskedNode  # 2 obs, 96 bytes
  ├── ProductNode  # 2 obs, 16 bytes
  │     ├── a: ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes
  │     ╰── b: ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes
  ╰── Bool[1, 0]

julia> m1 = MaskedModel(ProductModel(a=ArrayModel(Dense(2, 2)), b=ArrayModel(Dense(2, 2))))
MaskedModel  # 4 arrays, 12 params, 208 bytes
  ╰── ProductModel ↦ identity
        ├── a: ArrayModel(Dense(2 => 2))  # 2 arrays, 6 params, 104 bytes
        ╰── b: ArrayModel(Dense(2 => 2))  # 2 arrays, 6 params, 104 bytes

julia> m1(mn)
4×2 Matrix{Float32}:
 -2.36404  -0.0
 -2.11369  -0.0
 -6.30888  -0.0
 -2.53672  -0.0
```

Note the second observation being nullified, as the mask said to use only the first observation

For more abstract structures use reflectinmodel
```jldoctest masked_model
julia> Random.seed!(0)

julia> n = ProductNode(x1 = ArrayNode(Flux.onehotbatch([1, 2], 1:2)),
            x2 = BagNode(ArrayNode([1 2; 3 4]), [1:2, 0:-1]),
            x3 = ArrayNode(rand(2, 2)))
ProductNode  # 2 obs, 40 bytes
├── x1: ArrayNode(2×2 OneHotArray with Bool elements)  # 2 obs, 80 bytes
├── x2: BagNode  # 2 obs, 96 bytes
│         ╰── ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes
╰── x3: ArrayNode(2×2 Array with Float64 elements)  # 2 obs, 80 bytes

julia> mn = MaskedNode(n)
MaskedNode  # 2 obs, 120 bytes
  ├── ProductNode  # 2 obs, 40 bytes
  │     ├── x1: ArrayNode(2×2 OneHotArray with Bool elements)  # 2 obs, 80 bytes
  │     ├── x2: BagNode  # 2 obs, 96 bytes
  │     │         ╰── ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes
  │     ╰── x3: ArrayNode(2×2 Array with Float64 elements)  # 2 obs, 80 bytes
  ╰── Bool[1, 1]

julia> model = reflectinmodel(mn)
MaskedModel  # 12 arrays, 640 params, 2.969 KiB
  ╰── ProductModel ↦ Dense(30 => 10)  # 2 arrays, 310 params, 1.289 KiB
        ├── x1: ArrayModel(Dense(2 => 10))  # 2 arrays, 30 params, 200 bytes
        ├── x2: BagModel ↦ BagCount([SegmentedMean(10); SegmentedMax(10)]) ↦ Dense(21 => 10)  # 4 arrays, ...
		│         ╰── ArrayModel(Dense(2 => 10))  # 2 arrays, 30 params, 200 bytes
        ╰── x3: ArrayModel(Dense(2 => 10))  # 2 arrays, 30 params, 200 bytes

julia> model(mn)
10×2 Matrix{Float32}:
 -0.928482  -0.476943
  1.96143    0.177565
 -0.43994   -0.0348646
  0.600319   0.350153
  0.70722    0.557332
 -0.466048  -0.0849207
  1.55897    0.231923
 -0.544745  -0.162593
  0.283407  -0.0806759
  0.330414  -0.323688
```

Note that without the mask specification, it is set to true for all observations.

See also: [`AbstractMillModel`](@ref), [`MaskedModel`](@ref), [`reflectinmodel`](@ref).
"""
struct MaskedModel{T} <: AbstractMillModel
    m::T
end

Flux.@functor MaskedModel

function (m::MaskedModel)(x::MaskedNode)
    m.m(x.data) .* x.mask'
end

import Mill: reflectinmodel, _reflectinmodel

function _reflectinmodel(x::AbstractMaskedNode, fm, fa, fsm, fsa, s, ski, args...)
    m, d = _reflectinmodel(x.data, fm, fa, fsm, fsa, s, ski, args...)
    MaskedModel(m), d
end
