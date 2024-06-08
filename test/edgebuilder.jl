using NeuroPlanner
using LinearAlgebra
using NeuroPlanner: EdgeBuilder, construct, FeaturedEdgeBuilder
using Test

@testset "EdgeBuilder" begin
    # nv = 7; max_edges = 5; arity = 2
    for (arity, max_edges, nv) in [(2, 5, 7), (3, 7, 5)]
        eb = EdgeBuilder(arity, max_edges, nv)
        edges = [tuple([rand(1:nv) for _ in 1:arity]...) for _ in 1:max_edges]
        foreach(e -> push!(eb, e), edges)

        @testset "correctness of construction" begin
            ds = construct(eb, :x)
            @test all(ds.data.data[i].data.ii == [e[i] for e in edges] for i in 1:arity)
            @test all(all(j ∈ ds.bags[e] for (j, e) in enumerate(ds.data.data[i].data.ii)) for i in 1:arity)
        end
    end

    @testset "Correctness of filling only part of the whole max_edges" begin
        for (arity, max_edges, nv, pushed) in [(2, 5, 7, 2), (3, 7, 5, 6), (2, 5, 7, 0), (3, 2, 4, 2)]
            eb = EdgeBuilder(arity, max_edges, nv)
            edges = [tuple([rand(1:nv) for _ in 1:arity]...) for _ in 1:max_edges]
            foreach(e -> push!(eb, e), edges[1:pushed])

            @testset "correctness of construction" begin
                ds = construct(eb, :x)
                @test all(ds.data.data[i].data.ii == [e[i] for e in edges[1:pushed]] for i in 1:arity)
                @test all(all(j ∈ ds.bags[e] for (j, e) in enumerate(ds.data.data[i].data.ii)) for i in 1:arity)
            end
        end
    end
end


@testset "FeaturedEdgeBuilder" begin
    nv = 7; max_edges = 13; arity = 2; num_features = 5
    feb = FeaturedEdgeBuilder(arity, max_edges, nv, num_features; agg = +)
    edges = [(1,2),(2,3),(3,4),(1,2),(3,4)]
    for (i,e) in enumerate(edges)
        x = NeuroPlanner.Flux.onehot(i,1:num_features)
        push!(feb, e, x)
    end
    ds = construct(feb, :x)
    @test ds.data.data[1].data.e == :x
    @test ds.data.data[1].data.ii == [1,2,3]
    @test ds.data.data[2].data.e == :x
    @test ds.data.data[2].data.ii == [2,3,4]
    @test ds.data.data[3].data == Float32[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0; 0.0 0.0 1.0]

    # Let's try empty edges
    feb = FeaturedEdgeBuilder(arity, max_edges, nv, num_features; agg = +)
    ds = construct(feb, :x)
    @test ds.data.data[1].data.e == :x
    @test ds.data.data[1].data.ii == Int[]
    @test ds.data.data[2].data.e == :x
    @test ds.data.data[2].data.ii == Int[]
    @test ds.data.data[3].data == Matrix{Float32}(undef, 5, 0)
    

    # Let's try the dege builder without the deduplication
    feb = FeaturedEdgeBuilder(arity, max_edges, nv, num_features;agg = nothing)
    edges = [(1,2),(2,3),(3,4),(1,2),(3,4)]
    for (i,e) in enumerate(edges)
        x = NeuroPlanner.Flux.onehot(i,1:num_features)
        push!(feb, e, x)
    end
    ds = construct(feb, :x)
    @test ds.data.data[1].data.e == :x
    @test ds.data.data[1].data.ii == [1,2,3,1,3]
    @test ds.data.data[2].data.e == :x
    @test ds.data.data[2].data.ii == [2,3,4,2,4]
    @test ds.data.data[3].data == LinearAlgebra.I
    
end

@testset "MultiEdgeBuilder" begin
    @test_broken "add some tests"
end
