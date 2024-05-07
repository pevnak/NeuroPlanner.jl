using NeuroPlanner

md2 = fill("metadata", 2)
md3 = fill("metadata", 3)
md4 = fill("metadata", 4)
a = BagNode(rand(3, 4), [1:4], md4)
b = BagNode(rand(3, 4), [1:2, 3:4], md4)
c = BagNode(rand(3, 4), [1:1, 2:2, 3:4], md4)
d = BagNode(rand(3, 4), [1:4, 0:-1], md4)
wa = WeightedBagNode(rand(3, 4), [1:4], rand(1:4, 4) .|> Float64, md4)
wb = WeightedBagNode(rand(3, 4), [1:2, 3:4], rand(1:4, 4) .|> Float64, md4)
wc = WeightedBagNode(rand(3, 4), [1:1, 2:2, 3:4], rand(1:4, 4) .|> Float64, md4)
wd = WeightedBagNode(rand(3, 4), [1:4, 0:-1], rand(1:4, 4) .|> Float64, md4)
e = ArrayNode(rand(2, 2), md2)

f = ProductNode((wb, b), md2)
g = ProductNode([c, wc], md3)
h = ProductNode((wc, c), md3)
i = ProductNode((
        b,
        ProductNode((
            b,
            BagNode(
                BagNode(
                    rand(2, 4),
                    [1:1, 2:2, 3:3, 4:4]
                ),
                [1:3, 4:4]
            )
        ))
    ), md2)
k = ProductNode((a=wb, b=b), md2)
l = ProductNode((a=wc, b=c), md3)
m = ProductNode((a=wc, c=c), md3)
n = ProductNode((a=c, c=wc), md3)

o = MaskedNode(f, BitVector(ones(numobs(f))))
p = MaskedNode(g, BitVector([0, 0, 0]))
q = MaskedNode(h)
r = MaskedNode(i, BitVector([1, 0]))
s = MaskedNode(k, BitVector(ones(numobs(k))))
t = MaskedNode(l, BitVector([1, 0, 0]))
u = MaskedNode(m)
v = MaskedNode(n, BitVector([1, 0, 1]))

@testset "masked node" begin
    @testset "constructor logic" begin
        x = randn(2, 2)
        n1 = ArrayNode(x)
        bs = [1:1, 2:2]
        b = bags(bs)
        w = [0.1, 0.2]

        for md in [tuple(), tuple(nothing), tuple("metadata")]
            n2 = ProductNode((a=n1, b=n1), md...)

            n3 = MaskedNode(n2, BitVector([1, 1]))
            @test n3 isa MaskedNode{typeof(n2),BitVector}
            @test n3 == MaskedNode(n2)

            n8 = MaskedNode(n3)
            @test n8 == MaskedNode(n3, BitVector(ones(numobs(n3))))
        end
    end

    @testset "constructor assertions" begin
        for (x, y) in [(g, i), (f, g), (k, m)]
            @test_throws AssertionError MaskedNode(x, BitVector(ones(Bool, numobs(y))))
            @test_throws AssertionError MaskedNode(y, BitVector(ones(Bool, numobs(x))))
        end
    end

    @testset "numobs" begin
        @test numobs(o) == numobs(f)
        @test numobs(p) == numobs(g)
        @test numobs(q) == numobs(h)
    end


    @testset "hierarchical catobs on MaskedNodes" begin
        @test catobs(o, q).data.data[1].data.data == reduce(catobs, [o, q]).data.data[1].data.data ==
              catobs(f, h).data[1].data.data

        @test catobs(o, q).data.data[2].data.data == reduce(catobs, [o, q]).data.data[2].data.data ==
              catobs(f, h).data[2].data.data

        @test catobs(o, q).mask == reduce(catobs, [o, q]).mask


        @test catobs(o, q, o).data.data[1].data.data == reduce(catobs, [o, q, o]).data.data[1].data.data ==
              catobs(f, h, f).data[1].data.data

        @test catobs(o, q, o).mask == reduce(catobs, [o, q, o]).mask

        @test numobs(catobs(o, q)) == numobs(o) + numobs(q)


        @test catobs(p, p).data.data[1].data.data == reduce(catobs, [p, p]).data.data[1].data.data ==
              catobs(g, g).data[1].data.data

        @test catobs(p, p).data.data[2].data.data == reduce(catobs, [p, p]).data.data[2].data.data ==
              catobs(g, g).data[2].data.data

        @test catobs(p, p).mask == reduce(catobs, [p, p]).mask
        @test numobs(catobs(p, p)) == 2numobs(p)
        @test catobs(p, p).mask == vcat(p.mask, p.mask)


        @test catobs(p, p, p).data.data[1].data.data == reduce(catobs, [p, p, p]).data.data[1].data.data ==
              catobs(g, g, g).data[1].data.data

        @test catobs(p, p, p).mask == reduce(catobs, [p, p, p]).mask


        @test catobs(s, t).data.data[1].data.data == hcat(wb.data.data, wc.data.data)
        @test catobs(s, t).data.data[2].data.data == hcat(b.data.data, c.data.data)
        @test numobs(catobs(s, t)) == numobs(s) + numobs(t)
        @test catobs(s, t).mask == vcat(s.mask, t.mask)


        # correct length/keyset but different subtrees
        @test_throws MethodError catobs(o, r)
        @test_throws MethodError reduce(catobs, [f, i])
        @test_throws MethodError catobs(u, v)
        @test_throws MethodError reduce(catobs, [u, v])

        # different tuple length or keyset
        @test_throws ArgumentError catobs(p, r)
        @test_throws ArgumentError reduce(catobs, [p, r])
        @test_throws ArgumentError catobs(o, p)
        @test_throws ArgumentError reduce(catobs, [o, p])
        @test_throws ArgumentError catobs(s, u)
        @test_throws ArgumentError reduce(catobs, [s, u])
        @test_throws ArgumentError catobs(t, u)
        @test_throws ArgumentError reduce(catobs, [t, u])
    end

    @testset "catobs stability" begin
        for n in [r, s, u]
            @inferred catobs(n, n)
            @inferred reduce(catobs, [n, n])
            @inferred catobs([n, n])
        end
    end

    @testset "equals and hash" begin
        o2 = deepcopy(o)
        q2 = deepcopy(q)

        @test o ≠ q
        @test o ≠ q2
        @test o == o2
        @test q == q2

        @test hash(q) ≡ hash(q2)
        @test hash(o) ≡ hash(o2)
        @test hash(o) ≢ hash(q)
    end

    @testset "equals with missings" begin
        j = MaskedNode(f)
        q = MaskedNode(g)
        @test f ≠ g
        @test j ≠ q
        @test isequal(j, j)
        @test !isequal(j, q)
    end
end
