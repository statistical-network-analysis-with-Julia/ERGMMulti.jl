using ERGMMulti
using ERGM
using Network
using Random
using Statistics
using Test

# Two-layer directed fixture on 4 actors:
# friendship: 1→2, 2→1, 1→3, 3→4
# advice:     1→2, 2→3, 3→4, 4→3
function fixture()
    m = MultilayerNetwork(4; directed=true)
    add_layer!(m, :friendship)
    add_layer!(m, :advice)
    for (i, j) in [(1, 2), (2, 1), (1, 3), (3, 4)]
        add_layer_edge!(m, :friendship, i, j)
    end
    for (i, j) in [(1, 2), (2, 3), (3, 4), (4, 3)]
        add_layer_edge!(m, :advice, i, j)
    end
    return m
end

# Brute-force change stat: toggle (l, i, j) on a deep copy and recompute
function brute_change(term, m, l, i, j)
    nets = [ERGMMulti._copy_net(net) for net in m.layers]
    work = as_multilayer(nets, layer_names(m))
    net_l = work.layers[l]
    had = has_edge(net_l, i, j)
    had && rem_edge!(net_l, i, j)
    s0 = compute(term, work)
    add_edge!(net_l, i, j)
    s1 = compute(term, work)
    return s1 - s0
end

@testset "ERGMMulti.jl" begin
    @testset "MultilayerNetwork structure" begin
        m = fixture()
        @test ERGMMulti.n_layers(m) == 2
        @test layer_names(m) == [:friendship, :advice]
        @test ne(layer_network(m, :friendship)) == 4
        @test ne(layer_network(m, 2)) == 4
        @test_throws ArgumentError layer_network(m, :bogus)
        @test_throws ArgumentError add_layer!(m, :advice)

        # Mismatched layer network rejected
        @test_throws ArgumentError add_layer!(m, :x; net=network(3))
        @test_throws ArgumentError add_layer!(m, :y; net=network(4; directed=false))
    end

    @testset "Block-diagonal combined network" begin
        m = fixture()
        c = combine_networks(m)

        @test nv(c) == 8  # 4 actors × 2 layers
        @test ne(c) == 8  # 4 + 4 edges

        # Layer and actor membership attributes
        @test get_vertex_attribute(c, :layer, 3) == 1
        @test get_vertex_attribute(c, :layer, 7) == 2
        @test get_vertex_attribute(c, :actor, 7) == 3

        # Edges placed in the right blocks
        @test has_edge(c, 1, 2)      # friendship 1→2 in block 1
        @test has_edge(c, 5, 6)      # advice 1→2 in block 2
        @test has_edge(c, 8, 7)      # advice 4→3
        @test !has_edge(c, 1, 6)     # no cross-block edges

        # Round trip
        m2 = split_by_layer(c, 4, 2; names=[:friendship, :advice])
        @test ne(layer_network(m2, :friendship)) == 4
        @test has_edge(layer_network(m2, :advice), 4, 3)
    end

    @testset "Term values on the fixture" begin
        m = fixture()

        @test compute(LayerEdges(1), m) == 4.0
        @test compute(LayerEdges(2), m) == 4.0
        @test compute(LayerEdges(), m) == 8.0
        @test compute(LayerEdges([1, 2]), m) == 8.0

        @test compute(LayerMutual(1), m) == 1.0   # 1↔2 in friendship
        @test compute(LayerMutual(2), m) == 1.0   # 3↔4 in advice
        @test compute(LayerMutual(), m) == 2.0

        # Co-occurrence: 1→2 and 3→4 are in both layers
        @test compute(InterlayerDependence(1, 2), m) == 2.0

        # Cross reciprocity friendship→advice: (2,1): 2→1 in f, 1→2 in a ✓;
        # (1,2): 2→1 in a? no; (3,4): 4→3 in a ✓ → 2
        @test compute(MultiplexMutual(1, 2), m) == 2.0

        # WithinLayer lifts ERGM terms
        @test compute(WithinLayer(Edges(), 1), m) == 4.0
        @test compute(WithinLayer(Mutual(), 2), m) == 1.0
        @test compute(LayerTriangle(), m) ==
              compute(WithinLayer(Triangle(), 1), m) +
              compute(WithinLayer(Triangle(), 2), m)

        @test_throws ArgumentError InterlayerDependence(1, 1)
        @test_throws ArgumentError MultiplexMutual(2, 2)
    end

    @testset "change_stat_layer matches brute force" begin
        m = fixture()
        terms = [LayerEdges(1), LayerEdges(), LayerMutual(),
                 LayerTriangle(2), WithinLayer(Triangle(), 1),
                 InterlayerDependence(1, 2), MultiplexMutual(1, 2),
                 WithinLayer(TwoPath(), 2)]

        for term in terms, l in 1:2, i in 1:4, j in 1:4
            i == j && continue
            expected = brute_change(term, m, l, i, j)
            actual = change_stat_layer(term, m, l, i, j)
            @test actual ≈ expected atol = 1e-10
        end
    end

    @testset "MPLE: per-layer and pooled edges (analytic)" begin
        m = fixture()
        n_dyads = 4 * 3  # directed dyads per layer

        # Per-layer edges-only fits give logit of each layer's density
        r = ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
        @test r.converged
        d = 4 / n_dyads
        @test r.coefficients[1] ≈ log(d / (1 - d)) atol = 1e-5
        @test r.coefficients[2] ≈ log(d / (1 - d)) atol = 1e-5

        # Pooled fit gives logit of the pooled density
        rp = ergm_multi(m, [LayerEdges()])
        dp = 8 / (2 * n_dyads)
        @test rp.coefficients[1] ≈ log(dp / (1 - dp)) atol = 1e-5
        @test isfinite(rp.loglik) && rp.loglik < 0
        @test isfinite(rp.aic)
    end

    @testset "Offsets" begin
        m = fixture()
        # Fix the pooled edges coefficient; estimate the dependence term
        c_fix = -1.0
        r = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                       offsets=Dict(1 => c_fix))
        @test r.coefficients[1] == c_fix
        @test isnan(r.std_errors[1])
        @test isfinite(r.coefficients[2])
        @test !isnan(r.std_errors[2])

        @test_throws ArgumentError ergm_multi(m, [LayerEdges()];
                                              offsets=Dict(1 => 0.0))
        @test_throws ArgumentError ergm_multi(m, [LayerEdges()];
                                              offsets=Dict(5 => 0.0))
    end

    @testset "Simulation targets the model" begin
        rng = Random.Xoshiro(11)
        m = MultilayerNetwork(8; directed=true)
        add_layer!(m, :a)
        add_layer!(m, :b)

        θ_edges = -1.5
        θ_dep = 2.0
        terms = [LayerEdges(), InterlayerDependence(1, 2)]

        sims_dep = simulate_multi_ergm(m, terms, [θ_edges, θ_dep];
                                       n_sim=25, burnin=3000, interval=200, rng=rng)
        sims_ind = simulate_multi_ergm(m, terms, [θ_edges, 0.0];
                                       n_sim=25, burnin=3000, interval=200, rng=rng)

        dep_on = mean(compute(InterlayerDependence(1, 2), s) for s in sims_dep)
        dep_off = mean(compute(InterlayerDependence(1, 2), s) for s in sims_ind)
        @test dep_on > dep_off

        # Independent-layer model: each dyad Bernoulli(σ(θ_edges))
        p_edge = 1 / (1 + exp(-θ_edges))
        mean_edges = mean(compute(LayerEdges(), s) for s in sims_ind)
        @test mean_edges ≈ 2 * 56 * p_edge rtol = 0.2
    end

    @testset "Estimation recovers simulated coefficients" begin
        rng = Random.Xoshiro(5)
        m = MultilayerNetwork(10; directed=true)
        add_layer!(m, :a)
        add_layer!(m, :b)

        θ_true = [-1.2, 1.5]
        terms = [LayerEdges(), InterlayerDependence(1, 2)]
        draws = simulate_multi_ergm(m, terms, θ_true;
                                    n_sim=1, burnin=20000, interval=1, rng=rng)

        r = ergm_multi(draws[1], terms)
        @test r.converged
        @test r.coefficients[1] ≈ θ_true[1] atol = 0.5
        @test r.coefficients[2] ≈ θ_true[2] atol = 0.7
        @test r.coefficients[2] > 0
    end

    @testset "Multilevel descriptives" begin
        # Level 1: 5 people in 2 orgs; level 2: 2 orgs
        people = network(5; directed=false)
        add_edge!(people, 1, 2)   # same org
        add_edge!(people, 2, 3)   # cross org
        add_edge!(people, 4, 5)   # same org
        orgs = network(2; directed=false)

        membership = [Dict(1 => 1, 2 => 1, 3 => 2, 4 => 2, 5 => 2)]
        ml = MultilevelNetwork([people, orgs], membership)

        @test compute(LevelHomophily(1), ml) == 2.0   # (1,2) and (4,5)
        @test compute(Nestedness(1), ml) ≈ 2 / 3
        @test compute(CrossLevelEdge(), ml) == 0.0

        add_cross_level_edge!(ml, 1, 3, 2, 1)
        @test compute(CrossLevelEdge(), ml) == 1.0

        # Level with no parent errors clearly (formerly a BoundsError)
        @test_throws ArgumentError compute(LevelHomophily(2), ml)
        @test_throws ArgumentError add_cross_level_edge!(ml, 1, 1, 5, 1)
    end

    @testset "MultiNetwork" begin
        n1 = network(3)
        add_edge!(n1, 1, 2)
        n2 = network(4)
        add_edge!(n2, 1, 2)
        add_edge!(n2, 3, 4)
        mn = MultiNetwork([n1, n2], [:x, :y])
        @test length(mn) == 2
        @test compute(CrossNetEdges(), mn) == 3.0
    end

    @testset "Aliases" begin
        @test fit_multi_ergm === ergm_multi
    end
end
