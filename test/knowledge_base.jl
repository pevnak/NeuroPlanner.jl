using Test
using NeuroPlanner
using NeuroPlanner:  _catobs_kbs
using NeuroPlanner.Mill
using NeuroPlanner.Flux.Zygote


@testset "KnowledgeBase" begin 
	@testset "cat and getindex" begin 
		kb1 = KnowledgeBase((;a = randn(Float32, 3, 5), b = randn(Float32, 2, 4)))
		kb1 = append(kb1, :c, KBEntry(:a, [4,1,3,2]))
		kb1 = append(kb1, :d, KBEntry(:b, [4,1,3,2]))

		kb2 = KnowledgeBase((;a = randn(Float32, 3, 3), b = randn(Float32, 2, 5)))
		kb2 = append(kb2, :c, KBEntry(:a, [2,1,2]))
		kb2 = append(kb2, :d, KBEntry(:b, [3,1,4,5]))

		kbb = _catobs_kbs([kb1,kb2])
		@test kbb[:a] ≈ hcat(kb1[:a], kb2[:a])
		@test kbb[:b] ≈ hcat(kb1[:b], kb2[:b])
		@test Matrix(kbb, kbb[:c]) ≈ hcat(Matrix(kb1, kb1[:c]), Matrix(kb2, kb2[:c]))
		@test Matrix(kbb, kbb[:d]) ≈ hcat(Matrix(kb1, kb1[:d]), Matrix(kb2, kb2[:d]))

		@testset "ArrayNode" begin 
			kb1 = KnowledgeBase((;a = randn(Float32, 3,5), b = randn(Float32, 2,4)))
			kb1 = append(kb1, :c, ArrayNode(KBEntry(:a, [4,1,3,2])))
			kb1 = append(kb1, :d, ArrayNode(KBEntry(:b, [4,1,3,2])))

			kb2 = KnowledgeBase((;a = randn(Float32, 3,3), b = randn(Float32, 2,5)))
			kb2 = append(kb2, :c, ArrayNode(KBEntry(:a, [2,1,2])))
			kb2 = append(kb2, :d, ArrayNode(KBEntry(:b, [3,1,4,5])))

			kbb = _catobs_kbs([kb1,kb2])
			@test Matrix(kbb, kbb[:c].data) ≈ hcat(Matrix(kb1, kb1[:c].data), Matrix(kb2, kb2[:c].data))
			@test Matrix(kbb, kbb[:d].data) ≈ hcat(Matrix(kb1, kb1[:d].data), Matrix(kb2, kb2[:d].data))
		end


		@testset "ProductNode" begin 
			kb1 = KnowledgeBase((;a = randn(Float32, 3,7), b = randn(Float32, 2,4)))
			c1 = ArrayNode(KBEntry(:a, [4,1,3,2]))
			d1 = ArrayNode(KBEntry(:b, [1,3,3,2]))
			kb1 = append(kb1, :c, ProductNode((c1, d1)))
			kb1 = append(kb1, :d, ProductNode((a = c1, b = d1)))

			kb2 = KnowledgeBase((;a = randn(Float32, 3,4), b = randn(Float32, 2,4)))
			c2 = ArrayNode(KBEntry(:a, [1,4,2,3]))
			d2 = ArrayNode(KBEntry(:b, [4,1,3,4]))
			kb2 = append(kb2, :c, ProductNode((c2, d2)))
			kb2 = append(kb2, :d, ProductNode((a = c2, b = d2)))


			kbb = _catobs_kbs([kb1,kb2])
			@test Matrix(kbb, kbb[:c].data[1].data) ≈ hcat(Matrix(kb1, kb1[:c].data[1].data), Matrix(kb2, kb2[:c].data[1].data))
			@test Matrix(kbb, kbb[:c].data[2].data) ≈ hcat(Matrix(kb1, kb1[:c].data[2].data), Matrix(kb2, kb2[:c].data[2].data))
			@test Matrix(kbb, kbb[:d].data[:a].data) ≈ hcat(Matrix(kb1, kb1[:d].data[:a].data), Matrix(kb2, kb2[:d].data[:a].data))
			@test Matrix(kbb, kbb[:d].data[:b].data) ≈ hcat(Matrix(kb1, kb1[:d].data[:b].data), Matrix(kb2, kb2[:d].data[:b].data))
		end

		@testset "BagNode" begin 
			kb1 = KnowledgeBase((;a = randn(Float32, 3,7), b = randn(Float32, 2,4)))
			c1 = ArrayNode(KBEntry(:a, [4,1,3,2]))
			kb1 = append(kb1, :c, BagNode(c1, [1:2,3:4]))

			kb2 = KnowledgeBase((;a = randn(Float32, 3,4), b = randn(Float32, 2,4)))
			c2 = ArrayNode(KBEntry(:a, [1,4,2,3]))
			kb2 = append(kb2, :c, BagNode(c2, [1:3,4:4]))


			kbb = _catobs_kbs([kb1,kb2])
			@test Matrix(kbb, kbb[:c].data.data) ≈ hcat(Matrix(kb1, kb1[:c].data.data), Matrix(kb2, kb2[:c].data.data))
		end

		@testset "MaskedNode" begin
			kb1 = KnowledgeBase((;a = randn(Float32, 3,7), b = randn(Float32, 2,4)))
			c1 = ArrayNode(KBEntry(:a, [4,1,3,2]))
			d1 = ArrayNode(KBEntry(:b, [1,3,3,2]))
			kb1 = append(kb1, :c, MaskedNode(ProductNode((c1, d1)), BitVector([1,1,0,1])))
			kb1 = append(kb1, :d, MaskedNode(ProductNode((a = c1, b = d1))))

			kb2 = KnowledgeBase((;a = randn(Float32, 3,4), b = randn(Float32, 2,4)))
			c2 = ArrayNode(KBEntry(:a, [1,4,2,3]))
			d2 = ArrayNode(KBEntry(:b, [4,1,3,4]))
			kb2 = append(kb2, :c, MaskedNode(ProductNode((c2, d2))))
			kb2 = append(kb2, :d, MaskedNode(ProductNode((a = c2, b = d2)), BitVector([1,0,0,1])))


			kbb = _catobs_kbs([kb1,kb2])
			@test Matrix(kbb, kbb[:c].data.data[1].data) ≈ hcat(Matrix(kb1, kb1[:c].data.data[1].data), Matrix(kb2, kb2[:c].data.data[1].data))
			@test Matrix(kbb, kbb[:c].data.data[2].data) ≈ hcat(Matrix(kb1, kb1[:c].data.data[2].data), Matrix(kb2, kb2[:c].data.data[2].data))
			@test Matrix(kbb, kbb[:d].data[:a].data) ≈ hcat(Matrix(kb1, kb1[:d].data[:a].data), Matrix(kb2, kb2[:d].data[:a].data))
			@test Matrix(kbb, kbb[:d].data[:b].data) ≈ hcat(Matrix(kb1, kb1[:d].data[:b].data), Matrix(kb2, kb2[:d].data[:b].data))
		end

		@testset "Replace" begin
			@test KBEntry(:a, [4,1,3,2]) == KBEntry(:a, [4,1,3,2]) 
			@test KBEntry(:a, [4,1,3,2]) != KBEntry(:b, [4,1,3,2]) 
			@test KBEntry(:a, [4,1,3,2]) != KBEntry(:a, [1,1,3,2]) 

			a = KBEntry(:a, [4,1,3,2])
			b = KBEntry(:b, [4,1,3,2])
			@test replace(a, :a => :b) == b
			@test replace(a, :c => :b) == a


			a = ArrayNode(KBEntry(:a, [4,1,3,2]))
			b = ArrayNode(KBEntry(:b, [4,1,3,2]))
			c = ArrayNode(KBEntry(:c, [4,1,3,2]))
			@test replace(a, :a => :b) == b
			@test replace(a, :c => :b) == a
			@test replace(BagNode(a,[1:4]), :a => :b) == BagNode(b, [1:4])
			@test replace(BagNode(a,[1:4]), :c => :b) == BagNode(a, [1:4])
			@test replace(ProductNode((a,b)), :a => :b) == ProductNode((b,b))
			@test replace(ProductNode((a,b)), :a => :c) == ProductNode((c,b))
			@test replace(ProductNode((;a,b)), :a => :b) == ProductNode((;a = b, b))
			@test replace(ProductNode((;a,b)), :a => :c) == ProductNode((;a = c, b))
		end

	end


	@testset "Mill piping" begin
		kb = KnowledgeBase((;a = randn(Float32, 3,4), b = randn(Float32, 5,4)))
		a = KBEntry(:a, [4,1,3,2])
		b = KBEntry(:b, [3,1,3,2])
		@testset "Just Matrix" begin
			m = ArrayModel(Dense(3,1))
			ds = ArrayNode(a)
			@test m(kb, ds) ≈ m.m(Matrix(kb, a))
			@test gradient(kb -> sum(sin.(Matrix(kb, a))), kb)[1][:kb][:a] ≈ gradient(x -> sum(sin.(x)), kb[:a])[1]
			@test gradient(kb -> sum(sin.(m(kb, ds))), kb)[1][:kb][:a] ≈  gradient(x -> sum(sin.(m.m(x))), kb[:a])[1]
		end

		@testset "Full stack " begin
			kb = KnowledgeBase((;a = randn(Float32, 3,4), b = randn(Float32, 5,4)))
			a = KBEntry(:a, [4,1,3,2])
			b = KBEntry(:b, [3,1,3,2])
			kb = NeuroPlanner.append(kb, :c, ProductNode((;a = ArrayNode(a), b = ArrayNode(b))))
			kb = NeuroPlanner.append(kb, :d, MaskedNode(ProductNode((;a = ArrayNode(a), b = ArrayNode(b)), BitVector([0, 1, 1, 0]))))
			m = reflectinmodel(kb)

			@test gradient(kb -> sum(sin.(m(kb))), kb) !== nothing
			@test gradient(m -> sum(sin.(m(kb))), m) !== nothing


			kb = NeuroPlanner.append(NeuroPlanner.atoms(kb), :c, BagNode(ArrayNode(a), [1:2, 3:4, 0:-1]))
			m = reflectinmodel(kb)
			@test gradient(kb -> sum(sin.(m(kb))), kb) !== nothing
			@test gradient(m -> sum(sin.(m(kb))), m) !== nothing


			kb = NeuroPlanner.append(NeuroPlanner.atoms(kb), :c, BagNode(ProductNode((;a = ArrayNode(a), b = ArrayNode(b))), [1:2, 3:4, 0:-1]))
			m = reflectinmodel(kb)
			@test gradient(kb -> sum(sin.(m(kb))), kb) !== nothing
			@test gradient(m -> sum(sin.(m(kb))), m) !== nothing
		end
	end

	@testset "Random catobs test" begin 
		for i in 1:10
			kbs = map(1:10) do _
				kb = KnowledgeBase((;a = randn(Float32, 3,7), b = randn(Float32, 2,4)))
				c = ArrayNode(KBEntry(:a, rand(1:7,4)))
				d = ArrayNode(KBEntry(:b, rand(1:4,4)))
				n = rand(0:4)
				bags = Mill.length2bags([n, 4 - n])
				append(kb, :d, BagNode(ProductNode((a = c, b = d)), bags))
				append(kb, :e, MaskedNode(ProductNode((a = c, b = d))))
			end
			m = reflectinmodel(kbs[1])
			@test reduce(hcat, map(m, kbs)) ≈ m(reduce(catobs, kbs))
		end
	end

	@testset "Deduplication" begin
		@testset "DeduplicatedMatrix" begin 
			kb = KnowledgeBase((;a = [1 1 2 2 3 3]))
			@test NeuroPlanner._deduplicate(kb, kb[:a])[1] == DeduplicatedMatrix(kb[:a])
			@test NeuroPlanner._deduplicate(kb, kb[:a])[2] == DeduplicatedMatrix(kb[:a]).ii
		end

		@testset "ArrayNode" begin 
			kb = KnowledgeBase((;a = Float32[1 1 2 2 3 3], b = ArrayNode(KBEntry(:a, 1:6))))
			m = reflectinmodel(kb, d -> Dense(d,10), SegmentedMean)
			dekb = NeuroPlanner.deduplicate(kb)
			@test dekb[:a] == DeduplicatedMatrix(kb[:a])
			@test dekb[:b].ii == [1,1,2,2,3,3]
			@test m(dekb) ≈ m(kb)
		end

		@testset "BagNode" begin 
			kb = KnowledgeBase((;a = Float32[1 1 2 2 3 3], b = BagNode(ArrayNode(KBEntry(:a, 1:6)), [1:2,3:3,4:4,5:6])))
			m = reflectinmodel(kb, d -> Dense(d,10), SegmentedMean)
			dekb = NeuroPlanner.deduplicate(kb)
			@test m(dekb) ≈ m(kb)
			@test dekb[:b] isa NeuroPlanner.DeduplicatingNode
			@test dekb[:b].ii == [1,2,2,3]
			@test _isapprox(gradient(model -> sum(sin.(model(kb))), m)[1], gradient(model -> sum(sin.(model(dekb))), m)[1])
		end

		@testset "ProductNode" begin 
			a = Float32[1 1 2 2 3 3]
			b = Float32[1 2 2 2 2 3]
			kb = KnowledgeBase((;a, b, c = ProductNode((KBEntry(:a, 1:6), KBEntry(:b, 1:6),))))
			m = reflectinmodel(kb, d -> Dense(d,10), SegmentedMean)
			dekb = NeuroPlanner.deduplicate(kb)
			@test m(dekb) ≈ m(kb)
			@test dekb[:c] isa NeuroPlanner.DeduplicatingNode
			@test dekb[:c].ii == [1,2,3,3,4,5]
			@test _isapprox(gradient(model -> sum(sin.(model(kb))), m)[1], gradient(model -> sum(sin.(model(dekb))), m)[1])
		end

		@testset "MaskedNode" begin 
			a = Float32[1 1 2 2 3 3]
			b = Float32[1 2 2 2 2 3]
			kb = KnowledgeBase((;a, b, c = MaskedNode(ProductNode((KBEntry(:a, 1:6), KBEntry(:b, 1:6),)), BitVector([1, 0, 1, 1, 0, 0]))))
			m = reflectinmodel(kb, d -> Dense(d,10), SegmentedMean)
			dekb = NeuroPlanner.deduplicate(kb)
			@test m(dekb) ≈ m(kb)
			@test dekb[:c] isa NeuroPlanner.DeduplicatingNode
			@test dekb[:c].ii == [1,2,3,3,4,5]
			@test _isapprox(gradient(model -> sum(sin.(model(kb))), m)[1], gradient(model -> sum(sin.(model(dekb))), m)[1])
		end

		@testset "More complicated problem" begin 
			a = Float32[1 1 2 2 3 3]
			b = Float32[1 1 2 2 3 3]
			kb = KnowledgeBase((;a, b, 
				c = ProductNode((KBEntry(:a, 1:6), KBEntry(:b, 1:6),)),
				d = BagNode(KBEntry(:c, [1,1,2,2,3,3]), [1:2,1:2,3:4,4:5,4:5]),
				e = MaskedNode(ProductNode((KBEntry(:a, 1:6), KBEntry(:b, 1:6),)), BitVector([1, 0, 1, 1, 0, 0]))
			))
			m = reflectinmodel(kb, d -> Dense(d,10), SegmentedMean)
			dekb = deduplicate(kb)
			@test m(dekb) ≈ m(kb)
			@test _isapprox(gradient(model -> sum(sin.(model(kb))), m)[1], gradient(model -> sum(sin.(model(dekb))), m)[1])
		end
	end
end
