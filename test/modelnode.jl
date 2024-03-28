# use only activations without "kinks" for numerical gradient checking
# see e.g. https://stackoverflow.com/questions/40623512/how-to-check-relu-gradient

ACTIVATIONS = [identity, σ, swish, softplus, logcosh, mish, tanhshrink, lisht]
LAYERBUILDER = k -> f64(Flux.Dense(k, 2, rand(ACTIVATIONS)))
ABUILDER = d -> BagCount(all_aggregations(Float64, d))


@testset "masked models" begin
    a = randn(3, 4)
    b = randn(4, 4)
    c = randn(3, 4)
    p1 = ProductNode(; a, b)
    p2 = ProductNode(; b, a)
    p3 = ProductNode(; a, b, c)

    x1 = MaskedNode(p1)
    x2 = MaskedNode(p2)
    x3 = MaskedNode(p3)

    m = reflectinmodel(x1, LAYERBUILDER)
    @test m isa MaskedModel
    @test m.m isa ProductModel
    @test m.m.ms[:a] isa ArrayModel
    @test m.m.ms[:b] isa ArrayModel

    @test m(x1) == m(x2) == m(x3)

    for x in [x1, x2, x3]
        @test m(x) == m.m.m(vcat(m.m.ms[:a].m(a),
            m.m.ms[:b].m(b)))
        @test eltype(m(x)) == Float64
        @test size(m(x)) == (2, 4)
        @inferred m(x)
    end

    a = BagNode(randn(3, 4), [1:2, 3:4])
    b = BagNode(randn(4, 4), [1:1, 2:4])
    c = BagNode(randn(2, 4), [1:1, 2:4])
    p1 = ProductNode((a, b))
    p2 = ProductNode((a, b, c))
    p3 = ProductNode(a)

    x1 = MaskedNode(p1)
    x2 = MaskedNode(p2)
    x3 = MaskedNode(p3)

    m = reflectinmodel(x1, LAYERBUILDER)
    @test m isa MaskedModel
    @test m.m isa ProductModel
    @test m.m.ms[1] isa BagModel
    @test m.m.ms[1].im isa ArrayModel
    @test m.m.ms[1].bm isa Dense
    @test m.m.ms[2] isa BagModel
    @test m.m.ms[2].im isa ArrayModel
    @test m.m.ms[2].bm isa Dense

    ma = m.m.ms[1]
    mb = m.m.ms[2]
    for x in [x1, x2]
        @test m(x) == m.m.m(vcat(ma.bm(ma.a(ma.im.m(a.data.data), a.bags)),
            mb.bm(mb.a(mb.im.m(b.data.data), b.bags))))
        @test size(m(x)) == (2, 2)
        @test eltype(m(x)) == Float64
        @inferred m(x)
    end
    @test_throws AssertionError m(x3)

    p = ProductNode([randn(3, 4)])
    x = MaskedNode(p)
    m = reflectinmodel(x, LAYERBUILDER)
    @test m isa MaskedModel
    @test m.m isa ProductModel
    @test m.m.ms[1] isa ArrayModel

    @test size(m(x)) == (2, 4)
    @test eltype(m(x)) == Float64

    @test m(x) == m.m.m(m.m.ms[1].m(x.data.data[1].data))
end


@testset "catobs" begin
    b1 = BagNode(BagNode(randn(Float64, 2, 2), [1:2]), [1:1])
    b2 = BagNode(missing, [0:-1])
    b3 = BagNode(b1.data[1:0], [0:-1])

    p1 = ProductNode(b1)
    p2 = ProductNode(b2)
    p3 = ProductNode(b3)

    a = MaskedNode(p1)
    b = MaskedNode(p2)
    c = MaskedNode(p3)

    abc = catobs(a, b, c)
    bca = catobs(b, c, a)

    m = reflectinmodel(a, LAYERBUILDER)

    ma = m(a)
    mb = m(b)
    mc = m(c)

    mabc = m(abc)
    mbca = m(bca)

    @test mb ≈ mc
    @test mabc[:, 1] ≈ ma
    @test mabc[:, 2] ≈ mb
    @test mabc[:, 3] ≈ mc
    @test mbca[:, 1] ≈ mb
    @test mbca[:, 2] ≈ mc
    @test mbca[:, 3] ≈ ma

    @test m(b) == m.m.ms[1].bm(m.m.ms[1].a(b.data.data[1].data, b.data.data[1].bags))
    @test eltype(m(b)) == Float64
    @inferred m(b)
    for ds in [a, c, abc, bca]
        @test m(ds) == m.m.ms[1].bm(m.m.ms[1].a(m.m.ms[1].im.bm(m.m.ms[1].im.a(m.m.ms[1].im.im.m(ds.data.data[1].data.data.data), ds.data.data[1].data.bags)), ds.data.data[1].bags))
        @test eltype(m(ds)) == Float64
        @inferred m(ds)
    end
end
