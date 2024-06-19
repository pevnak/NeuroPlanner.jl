abstract type AbstractEdgeBuilder end
"""
    EdgeBuilder

    Simplifies construction of edges in the graph. The graph is assumed to be undirected.
    The idea is that at the start, we know the maximal number of edges and vertices in the graph.
    The constructor allocates approximate space. Then we can add edges one by one using the `push!.`
    Finally, when done, we `construct`, returns the computational graph.


Example of use
```julia
nv = 7          # number of vertices in the graph
capacity = 5    # number of edges
arity = 2       # arity of the edge

eb = EdgeBuilder(arity, capacity, nv)
push!(eb, (1, 2))   # add edge (1, 2)
push!(eb, (2, 3))   # add edge (2, 3)

ds = construct(eb, :x)  # returns the computational graph pointing to `:x` in knowledge graph
BagNode  # 7 obs, 304 bytes
  ╰── ProductNode  # 2 obs, 32 bytes
        ├── ArrayNode(Colon()×2 KBEntry with Float32 elements)  # 2 obs, 88 bytes
        ╰── ArrayNode(Colon()×2 KBEntry with Float32 elements)  # 2 obs, 88 bytes
```    
"""
mutable struct EdgeBuilder{N, T<:Integer} <: AbstractEdgeBuilder
    indices::NTuple{N,Vector{T}}
    num_vertices::Int
    max_edges::Int
    num_edges::Int
    function EdgeBuilder(indices::NTuple{N,Vector{T}}, num_vertices::Int, max_edges::Int) where {N,T}
        num_edges = 0
        new{N,T}(indices, num_vertices, max_edges, num_edges)
    end
end

function EdgeBuilder(ar::Val{arity}, max_edges::Int, num_vertices::Int) where {arity}
    indices = ntuple(_ -> Vector{Int}(undef, max_edges), arity)
    EdgeBuilder(indices, num_vertices, max_edges)
end


EdgeBuilder(arity::Int, max_edges::Int, num_vertices::Int) = EdgeBuilder(Val(arity), max_edges, num_vertices)

function Base.push!(eb::EdgeBuilder, vertices::NTuple{N,I}) where {N,I<:Integer}
    @assert all(v <= eb.num_vertices for v in vertices) "Cannot push edge connected to non-existant vertex!"
    eb.num_edges += 1
    _mapenumerate_tuple(vertices) do i, vᵢ
        eb.indices[i][eb.num_edges] = vᵢ
    end
end

function construct_hyperedge(eb::EdgeBuilder, input_sym::Symbol)
    indices = map(ii -> ii[1:eb.num_edges], eb.indices)
    xs = map(ii -> ArrayNode(KBEntry(input_sym, ii)), indices)
    # indices = view(eb.indices, :, 1:eb.num_edges)
    # xs = Tuple([ArrayNode(KBEntry(input_sym, indices[i, :])) for i in 1:eb.arity])
    CompressedBagNode(ProductNode(xs), CompressedBags(indices, eb.num_vertices, eb.num_edges))
end

function construct_gnnedge(eb::EdgeBuilder{2}, input_sym::Symbol)
    counts = zeros(Int, eb.num_vertices)
    @inbounds for i in 1:eb.num_edges
        counts[eb.indices[1][i]] += 1
        counts[eb.indices[2][i]] += 1
    end

    ends = cumsum(counts)
    start = ends .- (counts .- 1)
    bags = map(UnitRange, start, ends)

    indices = Vector{Int}(undef, 2*eb.num_edges)
    @inbounds for i in 1:eb.num_edges
        kᵢ = eb.indices[1][i]
        kⱼ = eb.indices[2][i]

        indices[start[kᵢ]] = kⱼ
        start[kᵢ] += 1

        indices[start[kⱼ]] = kᵢ
        start[kⱼ] += 1
    end

    BagNode(
        ArrayNode(KBEntry(input_sym, indices)),
        AlignedBags(bags)
    )
end

construct(eb::EdgeBuilder, input_sym::Symbol) = error("We do not implement construct for EdgeBuilder, because it is not meant for end-use. Edges are constructed either by `construct_gnnedge` or `construct_hyperedge`.")

struct HyperEdgeBuilder{EB<:EdgeBuilder} <: AbstractEdgeBuilder
    eb::EB
end

HyperEdgeBuilder(args...) = HyperEdgeBuilder(EdgeBuilder(args...))
Base.push!(eb::HyperEdgeBuilder, args...) = push!(eb.eb, args...)
construct(eb::HyperEdgeBuilder, input_sym) = construct_hyperedge(eb.eb, input_sym)

struct GnnEdgeBuilder{EB<:EdgeBuilder{2}} <: AbstractEdgeBuilder
    eb::EB
end

GnnEdgeBuilder(args...) = GnnEdgeBuilder(EdgeBuilder(args...))
function Base.push!(eb::GnnEdgeBuilder, e::Tuple{<:Integer,<:Integer}) 
    (i, j) = e
    e = i < j ? (i, j) : (j, i)
    push!(eb.eb, e)
end
construct(eb::GnnEdgeBuilder, input_sym) = construct_gnnedge(eb.eb, input_sym)

"""
    struct FeaturedEdgeBuilder{EB<:AbstractEdgeBuilder,M,F}
        eb::EB
        xe::M
        agg::F
    end


    Wraps `EdgeBuilder` and adds features on edges. The features on edges are mainly
    used to identify type of edge, as we want to compare between edges with features
    and multi-edges. When building the featured edges, there is also an option to 
    duplicate and deduplicate edges. When `agg = nothing,` edges are not deduplicated,
    which is faster during construction but it might be slower during inference. Conrary,
    when `agg=Function` (`+` is the default), the same edges are deduplicated and their 
    features are aggregated by `add`.
"""
struct FeaturedEdgeBuilder{EB<:AbstractEdgeBuilder,M,F}
    eb::EB
    xe::M
    agg::F
end

function FeaturedHyperEdgeBuilder(arity, max_edges::Int, num_vertices::Int, num_features::Int; agg=+)
    xe = zeros(Float32, num_features, max_edges)
    eb = HyperEdgeBuilder(arity, max_edges, num_vertices)
    FeaturedEdgeBuilder(eb, xe, agg)
end

function FeaturedGnnEdgeBuilder(arity, max_edges::Int, num_vertices::Int, num_features::Int; agg=+)
    xe = zeros(Float32, num_features, max_edges)
    eb = GnnEdgeBuilder(arity, max_edges, num_vertices)
    FeaturedEdgeBuilder(eb, xe, agg)
end

"""
    FeaturedGnnEdgeBuilderNA(arity::Int, max_edges::Int, num_vertices::Int, num_features::Int)

    construct a `FeaturedEdgeBuilder` without aggregation
"""
function FeaturedGnnEdgeBuilderNA(arity, max_edges::Int, num_vertices::Int, num_features::Int)
    FeaturedGnnEdgeBuilder(arity, max_edges, num_vertices, num_features;agg=nothing)    
end

"""
    FeaturedHyperEdgeBuilderNA(arity::Int, max_edges::Int, num_vertices::Int, num_features::Int)

    construct a `FeaturedEdgeBuilder` without aggregation
"""
function FeaturedHyperEdgeBuilderNA(arity, max_edges::Int, num_vertices::Int, num_features::Int)
    FeaturedHyperEdgeBuilder(arity, max_edges, num_vertices, num_features;agg=nothing)    
end

function Base.push!(feb::FeaturedEdgeBuilder, vertices::NTuple{N,I}, x) where {N,I<:Integer}
    eb = feb.eb.eb
    push!(eb, vertices)
    feb.xe[:, eb.num_edges] .= x
end

function Base.push!(feb::FeaturedEdgeBuilder, vertices::NTuple{N,I}, edge_type::Integer) where {N,I<:Integer}
    eb = feb.eb.eb
    push!(eb, vertices)
    feb.xe[edge_type, eb.num_edges] = 1
end

"""
    construct(feb::FeaturedEdgeBuilder, input_sym::Symbol)

    A version of FeaturedEdgeBuilder, where edges are deduplicated and their features are
    aggregated by `feb.agg`
"""
function construct(feb::FeaturedEdgeBuilder{<:Any,<:Any,<:Function}, input_sym::Symbol)
    eb = feb.eb.eb
    if eb.num_edges == 0 
        ds = _construct_featurededge(feb.eb, feb.xe, eb.indices, 0, eb.num_vertices, input_sym)
        return(ds)
    end
    x = @view feb.xe[:, 1:eb.num_edges]
    indices = map(ii -> (@view ii[1:eb.num_edges]), eb.indices)

    mask, ii = find_duplicates(indices...)
    dedupped_xe = NNlib.scatter(feb.agg, x, ii)
    dedupped_indices = map(ii -> ii[mask], indices)

    _construct_featurededge(feb.eb, dedupped_xe, dedupped_indices, length(dedupped_indices[1]), eb.num_vertices, input_sym)
end


"""
    construct(feb::FeaturedEdgeBuilder, input_sym::Symbol)

    if `agg` is nothing, then edges are not deduplicated
"""
function construct(feb::FeaturedEdgeBuilder{<:Any,<:Any,Nothing}, input_sym::Symbol)
    eb = feb.eb.eb
    _construct_featurededge(feb.eb, feb.xe, eb.indices, eb.num_edges, eb.num_vertices, input_sym)
end

function _construct_featurededge(::HyperEdgeBuilder, xe, indices, num_edges, num_vertices, input_sym::Symbol)
    x = xe[:, 1:num_edges]
    indices = map(ii -> ii[1:num_edges], indices)

    xs = map(Base.Fix1(KBEntry, input_sym), indices)
    xs = tuple(xs..., ArrayNode(x))
    CompressedBagNode(ProductNode(xs), CompressedBags(indices, num_vertices, num_edges))
end

function _construct_featurededge(::GnnEdgeBuilder, xe, indices, num_edges, num_vertices, input_sym::Symbol)
    # this is the part copied from the EdgeBuilder building the neighborhood
    counts = zeros(Int, num_vertices)
    @inbounds for i in 1:num_edges
        counts[indices[1][i]] += 1
        counts[indices[2][i]] += 1
    end

    ends = cumsum(counts)
    start = ends .- (counts .- 1)
    bags = map(UnitRange, start, ends)

    neighborhoods = Vector{Int}(undef, 2num_edges)
    x = similar(xe, size(xe, 1), 2num_edges) # this is the new part, where we construct features on edges
    @inbounds for i in 1:num_edges
        kᵢ = indices[1][i]
        kⱼ = indices[2][i]

        neighborhoods[start[kᵢ]] = kⱼ
        x[:, start[kᵢ]] .= @view xe[:, i] # this is where we construct edge features
        start[kᵢ] += 1

        neighborhoods[start[kⱼ]] = kᵢ
        x[:, start[kⱼ]] .= @view xe[:, i] # this is where we construct edge features
        start[kⱼ] += 1
    end

    BagNode(
        ProductNode((xe = ArrayNode(x), xv = ArrayNode(KBEntry(input_sym, neighborhoods)))),
        AlignedBags(bags)
    )
end


"""
    struct MultiEdgeBuilder{EB<:AbstractEdgeBuilder,T}
        eb::EB
        xe::Matrix{T}
    end

    Wraps `EdgeBuilder` and adds features on edges. When edges are constructed
    edges are duplicated and their feaures are summed. The `Matrix` for features
    one edges are duplicated. 
"""
struct MultiEdgeBuilder{N, EBS<:NTuple{N,<:AbstractEdgeBuilder}}
    ebs::EBS
end

function MultiHyperEdgeBuilder(arity, max_edges::Int, num_vertices::Int, num_features::Int)
    ebs = ntuple(_ -> HyperEdgeBuilder(arity, max_edges, num_vertices) ,num_features)
    MultiEdgeBuilder(ebs)
end

function MultiGnnEdgeBuilder(arity, max_edges::Int, num_vertices::Int, num_features::Int)
    ebs = tuple([GnnEdgeBuilder(arity, max_edges, num_vertices) for _ in 1:num_features]...)
    MultiEdgeBuilder(ebs)
end

function Base.push!(feb::MultiEdgeBuilder, vertices::NTuple{N,I}, edge_type::Integer) where {N,I<:Integer}
    push!(feb.ebs[edge_type], vertices)
end

function construct(feb::MultiEdgeBuilder, input_sym::Symbol)
    ProductNode(map(Base.Fix2(construct, input_sym), feb.ebs))
end