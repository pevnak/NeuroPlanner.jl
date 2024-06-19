using NeuroPlanner
using LinearAlgebra
using NeuroPlanner: FeaturedHyperEdgeBuilder, FeaturedGnnEdgeBuilder, EdgeBuilder, HyperEdgeBuilder, GnnEdgeBuilder, construct, construct_hyperedge, construct_gnnedge, FeaturedEdgeBuilder, MultiEdgeBuilder
using Test

@testset "HyperEdgeBuilder" begin
    # nv = 7; max_edges = 5; arity = 2
    for (arity, max_edges, nv) in [(2, 5, 7), (3, 7, 5)]
        for inserted_edges in 0:max_edges # we want to test the whole range of possible edges
            for (EBuilder, construct_fun) in ((EdgeBuilder, construct_hyperedge), (HyperEdgeBuilder, construct))
                eb = EBuilder(arity, max_edges, nv)
                edges = [tuple([rand(1:nv) for _ in 1:arity]...) for _ in 1:inserted_edges]
                foreach(e -> push!(eb, e), edges)

                @testset "correctness of construct_hyperion" begin
                    ds = construct_fun(eb, :x)
                    @test all(ds.data.data[i].data.ii == [e[i] for e in edges] for i in 1:arity)
                    @test all(all(j ∈ ds.bags[e] for (j, e) in enumerate(ds.data.data[i].data.ii)) for i in 1:arity)
                end
            end
        end
    end
end

"""
    slow_neighborhood(edges, nv)

    reference slow implementation of the neighborhood to verify the correctness
"""
function slow_neighborhood(edges, nv)
    neighborhoods = [Int[] for _ in 1:nv]
    for (i,j) in edges
        push!(neighborhoods[i], j)
        push!(neighborhoods[j], i)
    end
    map(sort, neighborhoods)
end

@testset "GnnEdgeBuilder" begin
    nv = 7; max_edges = 5; arity = 2

    # I want to construct_hyper gnn edges
    for inserted_edges in 0:max_edges
        for (EBuilder, construct_fun) in ((EdgeBuilder, construct_gnnedge), (GnnEdgeBuilder, construct))
            edges = [tuple([rand(1:nv) for _ in 1:arity]...) for _ in 1:inserted_edges]
            neighborhoods = slow_neighborhood(edges, nv)

            eb = EBuilder(arity, max_edges, nv)
            foreach(e -> push!(eb, e), edges)
            ds = construct_fun(eb, :x)
            ii = ds.data.data.ii 
            @test length(ds.bags) == nv
            @test all(sort(ii[ds.bags[i]]) == neighborhoods[i] for i in 1:nv)
        end
    end
end

@testset "FeaturedHyperEdgeBuilder" begin
    nv = 7; max_edges = 13; arity = 2; num_features = 5
    @testset "ith deduplication " begin 
        for inserted_edges in 0:max_edges
            feb = FeaturedHyperEdgeBuilder(arity, max_edges, nv, num_features;agg = +)
            edges = [tuple([rand(1:nv) for _ in 1:arity]...) for _ in 1:inserted_edges]
            edgemap = Dict{Tuple{Int,Int}, Vector{Float32}}()
            for (i,e) in enumerate(edges)
                x = NeuroPlanner.Flux.onehot(rand(1:num_features),1:num_features)
                push!(feb, e, x)
                edgemap[e] = get!(edgemap, e, zeros(Float32, num_features)) + x
            end
            ds = construct(feb, :x)
            xe = ds.data.data[end].data

            @test size(xe,2) == length(edgemap)
            @test length(ds.bags) == nv

            for (k, (i,j)) in enumerate(zip(ds.data.data[1].data.ii,  ds.data.data[2].data.ii))
                e = (i,j)
                x = xe[:, k] 
                @test e ∈ edges
                @test x == edgemap[e]
                @test k ∈ ds.bags[i]
                @test k ∈ ds.bags[j]
            end
        end
    end

    # Let's try the dege builder without the deduplication
    @testset "without deduplication " begin 
        for inserted_edges in 0:max_edges
            feb = FeaturedHyperEdgeBuilder(arity, max_edges, nv, num_features;agg = nothing)
            edges = [tuple(sort([rand(1:nv) for _ in 1:arity])...) for _ in 1:inserted_edges]
            edgemap = Dict{Tuple{Int,Int}, Any}()
            for (i,e) in enumerate(edges)
                x = NeuroPlanner.Flux.onehot(rand(1:num_features),1:num_features)
                push!(feb, e, x)
                push!(get!(edgemap, e, []), x)
            end
            ds = construct(feb, :x)
            xe = ds.data.data[end].data

            @test size(xe,2) == inserted_edges
            @test length(ds.bags) == nv

            for (k, (i,j)) in enumerate(zip(ds.data.data[1].data.ii,  ds.data.data[2].data.ii))
                e = (i,j)
                x = xe[:, k] 
                @test e ∈ edges
                @test x ∈ edgemap[e]
                @test k ∈ ds.bags[i]
                @test k ∈ ds.bags[j]
            end
        end
    end
end

@testset "FeaturedGnnEdgeBuilder" begin
    nv = 7; max_edges = 13; arity = 2; num_features = 5
    @testset "with deduplication " begin 
        for inserted_edges in 0:max_edges
            feb = FeaturedGnnEdgeBuilder(arity, max_edges, nv, num_features;agg = +)
            edges = [tuple(sort([rand(1:nv) for _ in 1:arity])...) for _ in 1:inserted_edges]
            edgemap = Dict{Tuple{Int,Int}, Vector{Float32}}()
            for (i,e) in enumerate(edges)
                x = NeuroPlanner.Flux.onehot(rand(1:num_features),1:num_features)
                push!(feb, e, x)
                edgemap[e] = get!(edgemap, e, zeros(Float32, num_features)) + x
            end
            ds = construct(feb, :x)
            xe = ds.data.data.xe.data

            @test size(xe,2) == 2*length(edgemap)
            @test length(ds.bags) == nv

            for (i, bag) in enumerate(ds.bags)
                for (k,j) in zip(bag, ds.data.data.xv.data.ii[bag])
                    e = i < j ? (i,j) : (j,i) 
                    x = xe[:, k] 
                    @test e ∈ edges
                    @test x == edgemap[e]
                end
            end
        end
    end

    # Let's try the dege builder without the deduplication
    @testset "without deduplication " begin 
        for inserted_edges in 0:max_edges
            feb = FeaturedGnnEdgeBuilder(arity, max_edges, nv, num_features;agg = nothing)
            edges = [tuple(sort([rand(1:nv) for _ in 1:arity])...) for _ in 1:inserted_edges]
            edgemap = Dict{Tuple{Int,Int}, Any}()
            for (i,e) in enumerate(edges)
                x = NeuroPlanner.Flux.onehot(rand(1:num_features),1:num_features)
                push!(feb, e, x)
                push!(get!(edgemap, e, []), x)
            end
            ds = construct(feb, :x)
            xe = ds.data.data.xe.data

            @test size(xe,2) == 2*length(edges)
            @test length(ds.bags) == nv

            for (i, bag) in enumerate(ds.bags)
                for (k,j) in zip(bag, ds.data.data.xv.data.ii[bag])
                    e = i < j ? (i,j) : (j,i) 
                    x = xe[:, k] 
                    @test e ∈ edges
                    @test x ∈ edgemap[e]
                end
            end
        end
    end
end

@testset "MultiEdgeBuilder" begin
    nv = 7; max_edges = 13; arity = 2; num_features = 2
    meb = MultiEdgeBuilder(arity, max_edges, nv, num_features)
    edges = [(1,2),(2,3),(3,4),(1,2),(3,4)]
    for (i,e) in enumerate(edges)
        push!(meb, e, mod(i, num_features) + 1)
    end
    ds = construct(meb, :x)

    @test ds.data[1].data.data[1].data.e == :x
    @test ds.data[1].data.data[1].data.ii == [2,1]
    @test ds.data[1].data.data[2].data.e == :x
    @test ds.data[1].data.data[2].data.ii == [3,2]

    @test ds.data[2].data.data[1].data.e == :x
    @test ds.data[2].data.data[1].data.ii == [1,3,3]
    @test ds.data[2].data.data[2].data.e == :x
    @test ds.data[2].data.data[2].data.ii == [2,4,4]    
end
