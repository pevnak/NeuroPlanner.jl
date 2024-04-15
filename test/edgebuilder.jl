using NeuroPlanner: EdgeBuilder, construct
using Test


@testset "EdgeBuilder" begin 
	nv = 7
	capacity = 5
	arity = 2
	for (arity, capacity, nv) in [(2,5,7), (3,7,5)]
		eb = EdgeBuilder(arity, capacity, nv)

		@testset "Constructor" begin 
			@test length(eb.ii) == arity
			@test length(eb.bags) == nv
			@test eb.first_free == 1
			@test eb.nv == nv
		end

		edges = [tuple([rand(1:nv) for _ in 1:arity]...) for _ in 1:capacity]
		@testset "adding edges" begin 
			for (i, e) in enumerate(edges)
				push!(eb, e)
				@test all(eb.ii[j][i] == e[j] for j in 1:arity)   # edge was added to i-th place	
				@test all(eb.bags[e[j]][end] == i for j in 1:arity) 	  # index of edge was added to bags
				@test eb.first_free == i + 1		              # counter was incremented
			end
			e = tuple([rand(1:nv) for _ in 1:arity]...)
			@test_throws ErrorException push!(eb, e)
		end

		@testset "correctness of construction" begin 
			ds = construct(eb, :x)
			@test all(ds.data.data[i].data.ii == [e[i] for e in edges] for i in 1:arity)
			@test all(all(j ∈ ds.bags[e] for  (j, e) in enumerate(ds.data.data[i].data.ii)) for i in 1:arity)
		end
	end

end


@testset "EdgeBuilderComp" begin
    for (arity, capacity, nv) in [(2, 5, 7), (3, 7, 5)]
        eb = EdgeBuilder(arity, capacity, nv)
        edges = [tuple([rand(1:nv) for _ in 1:arity]...) for _ in 1:capacity]
        foreach(e -> push!(eb, e), edges)

        @testset "correctness of construction" begin
            ds = construct(eb, :x)
            @test all(ds.data.data[i].data.ii == [e[i] for e in edges] for i in 1:arity)
            @test all(all(j ∈ ds.bags[e] for (j, e) in enumerate(ds.data.data[i].data.ii)) for i in 1:arity)
        end
    end

    @testset "Correctness of filling only part of the whole capacity" begin
        for (arity, capacity, nv, pushed) in [(2, 5, 7, 2), (3, 7, 5, 6), (2, 5, 7, 0), (3, 2, 4, 2)]
            eb = EdgeBuilder(arity, capacity, nv)
            edges = [tuple([rand(1:nv) for _ in 1:arity]...) for _ in 1:capacity]
            foreach(e -> push!(eb, e), edges[1:pushed])

            @testset "correctness of construction" begin
                ds = construct(eb, :x)
                @test all(ds.data.data[i].data.ii == [e[i] for e in edges[1:pushed]] for i in 1:arity)
                @test all(all(j ∈ ds.bags[e] for (j, e) in enumerate(ds.data.data[i].data.ii)) for i in 1:arity)
            end
        end
    end
end
