using ERGMMulti
using ERGM
using Networks
using Graphs: src, dst
using Random
using Statistics
using StatsAPI: StatsAPI
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

# Two-layer UNDIRECTED fixture on 5 actors (the undirected-only terms —
# Kstar, GWDegree, Degree — are covered on this one):
# friendship: 1-2, 1-3, 2-3, 3-4, 4-5
# advice:     1-2, 2-4, 3-4, 3-5
function fixture_undirected()
    m = MultilayerNetwork(5; directed=false)
    add_layer!(m, :friendship)
    add_layer!(m, :advice)
    for (i, j) in [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
        add_layer_edge!(m, :friendship, i, j)
    end
    for (i, j) in [(1, 2), (2, 4), (3, 4), (3, 5)]
        add_layer_edge!(m, :advice, i, j)
    end
    return m
end

# The error message of a call that must throw, for `occursin` assertions
function errmsg(f)
    try
        f()
        return ""
    catch e
        return sprint(showerror, e)
    end
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

        # An out-of-range layer INDEX is an ArgumentError naming the layers,
        # like the Symbol form and the term validator — not a raw BoundsError
        @test_throws ArgumentError layer_network(m, 5)
        @test_throws ArgumentError layer_network(m, 0)
        msg = errmsg(() -> layer_network(m, 5))
        @test occursin("layer index 5 out of range", msg) && occursin(":friendship", msg) &&
              occursin(":advice", msg) && occursin("1:2", msg)
        @test_throws ArgumentError add_layer_edge!(m, 5, 1, 2)
        @test occursin("out of range", errmsg(() -> add_layer_edge!(m, 5, 1, 2)))

        # An actor id outside 1:n, or a self-loop on a loops=false layer, is
        # refused — never silently dropped (add_edge!'s `false` used to vanish:
        # a 0-based id gave a smaller network and a fit on it, with no sign)
        for (i, j) in ((1, 5), (0, 2), (5, 1), (-1, 2), (2, 2), (4, 4))
            @test_throws ArgumentError add_layer_edge!(m, :friendship, i, j)
            @test_throws ArgumentError add_layer_edge!(m, 2, i, j)
        end
        msg = errmsg(() -> add_layer_edge!(m, :friendship, 0, 2))
        @test occursin("actor ids must lie in 1:4", msg) && occursin("(0, 2)", msg) &&
              occursin("0-based", msg)
        msg = errmsg(() -> add_layer_edge!(m, 2, 3, 3))
        @test occursin("(3, 3) is a self-loop", msg) && occursin(":advice", msg) &&
              occursin("loops=true", msg)
        @test ne(layer_network(m, :friendship)) == 4 && ne(layer_network(m, :advice)) == 4
        # ... while a layer built with loops=true accepts the loop here (the
        # model constructor is where a loop is refused)
        ml = MultilayerNetwork(3; directed=true)
        add_layer!(ml, :a; net=network(3; directed=true, loops=true))
        @test add_layer_edge!(ml, :a, 2, 2) === ml && has_edge(layer_network(ml, :a), 2, 2)
        @test_throws ArgumentError MultiERGMModel([LayerEdges()], ml)

        # Mismatched layer network rejected
        @test_throws ArgumentError add_layer!(m, :x; net=network(3))
        @test_throws ArgumentError add_layer!(m, :y; net=network(4; directed=false))
        @test sprint(show, m) ==
              "MultilayerNetwork: 4 actors, 2 directed layer(s) (:friendship, :advice)"

        # A two-mode (bipartite) layer is refused — by `add_layer!` and hence by
        # `as_multilayer` — instead of entering the one-mode dyad universe,
        # where the MPLE would enumerate the impossible within-mode pairs as
        # observed non-ties (it used to: `LayerEdges` on two bipartite 6-actor
        # layers with 2 ties each gave logit(2/15), not logit(2/9))
        b = network(4; bipartite=2); add_edge!(b, 1, 3)
        @test is_two_mode(b)
        @test_throws ArgumentError add_layer!(m, :bip; net=b)
        msg = errmsg(() -> add_layer!(m, :bip; net=b))
        @test occursin("layer :bip is two-mode (bipartite)", msg) &&
              occursin("one-mode layers only", msg) && occursin("b1dspL", msg)
        @test_throws ArgumentError as_multilayer([b, network(4; bipartite=2)], [:a, :b])
        @test occursin("two-mode", errmsg(() -> as_multilayer([b, network(4; bipartite=2)], [:a, :b])))
        @test layer_names(m) == [:friendship, :advice]       # nothing was added
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
        # Kstar/GWDegree are undirected-only since ERGM 0.2 (as in R): on the
        # DIRECTED fixture the directed variants carry the same statistics
        terms = [LayerEdges(1), LayerEdges(), LayerMutual(),
                 LayerTriangle(2), WithinLayer(Triangle(), 1),
                 InterlayerDependence(1, 2), MultiplexMutual(1, 2),
                 WithinLayer(TwoPath(), 2), WithinLayer(OStar(2), 1),
                 WithinLayer(IStar(2), 1), WithinLayer(GWESP(0.5), 2),
                 WithinLayer(GWODegree(0.3), 1), WithinLayer(GWIDegree(0.3), 1)]

        for term in terms, l in 1:2, i in 1:4, j in 1:4
            i == j && continue
            expected = brute_change(term, m, l, i, j)
            actual = change_stat_layer(term, m, l, i, j)
            @test actual ≈ expected atol = 1e-10
        end

        # ... and the undirected-only terms stay covered on the UNDIRECTED
        # fixture (unordered dyads i < j)
        mu = fixture_undirected()
        @test !is_directed(mu)
        terms_u = [LayerEdges(2), LayerEdges(), LayerTriangle(),
                   WithinLayer(Triangle(), 1), InterlayerDependence(1, 2),
                   WithinLayer(Kstar(2), 1), WithinLayer(GWDegree(0.3), 1),
                   WithinLayer(Degree(1), 2), WithinLayer(GWESP(0.5), 2)]
        for term in terms_u, l in 1:2, i in 1:5, j in (i+1):5
            expected = brute_change(term, mu, l, i, j)
            actual = change_stat_layer(term, mu, l, i, j)
            @test actual ≈ expected atol = 1e-10
        end

        # The lifted term is validated against ITS layer: an undirected-only
        # term on a directed layer throws ERGM.jl's own error (with its hint)
        # instead of silently computing out-stars under the wrong label
        msg = errmsg(() -> ergm_multi(m, [WithinLayer(Kstar(2), 1)]))
        @test occursin("kstar2", msg) && occursin("OStar(2)", msg)
        @test_throws ArgumentError compute(WithinLayer(Kstar(2), 1), m)
        @test_throws ArgumentError change_stat_layer(WithinLayer(GWDegree(0.3), 1), m, 1, 1, 2)
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

    # ------------------------------------------------------------------
    # Allocation regression on the MPLE derivative loop (review finding 15)
    #
    # `_multi_mple_fit` used to carry its own logistic loop with a per-dyad
    # `(pr*(1-pr)) .* (x * x')` inside it: a fresh pf×pf matrix on every one of
    # the n_dyads rows of every Newton evaluation (649 KB per evaluation on a
    # 2-layer, 40-actor design). It now runs on the shared, workspace-backed
    # `ERGM.logistic_derivatives`. This test is what stops the outer product
    # coming back — and it measures the closure the FITTER builds, from the
    # package's own design matrix, not a copy of the loop.
    # ------------------------------------------------------------------
    @testset "MPLE derivative evaluations allocate O(p²), not O(n_dyads · p²)" begin
        terms = AbstractERGMTerm[LayerEdges(1), LayerEdges(2),
                                 WithinLayer(NodeMatch(:grp), 1)]

        function evaluation_allocs(n_actors)
            rng = Random.Xoshiro(5)
            mm = MultilayerNetwork(n_actors; directed=true)
            for l in 1:2
                net = network(n_actors; directed=true)
                for v in 1:n_actors
                    set_vertex_attribute!(net, :grp, v, isodd(v) ? "a" : "b")
                end
                for i in 1:n_actors, j in 1:n_actors
                    i != j && rand(rng) < 0.2 && add_edge!(net, i, j)
                end
                add_layer!(mm, Symbol("L", l); net=net)
            end
            model = MultiERGMModel(terms, mm)
            free = collect(1:length(model.terms))
            Xf, y, η0 = ERGMMulti._multi_mple_design(mm, model.terms,
                                                     model.offsets, free)
            d = Networks.logistic_derivatives(Xf, y; offset=η0)
            β = fill(0.1, length(free))
            d(β)                    # warm up: @allocated on a first call
            return size(Xf, 1), @allocated d(β)   # would measure compilation
        end

        rows_small, a_small = evaluation_allocs(8)     # 112 dyads
        rows_big, a_big = evaluation_allocs(40)        # 3120 dyads
        @test rows_big > 25 * rows_small
        # 28x the dyads, the same allocations: the workspaces are reused and only
        # the (p) gradient and (p×p) Hessian returned to `newton_fit` are new.
        @test a_small <= 512
        @test a_big <= 512
        @test a_big <= a_small + 64
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

    @testset "Simulation preserves vertex attributes (regression)" begin
        # Regression: _copy_net used to drop vertex attributes, so attribute
        # terms (e.g. WithinLayer(NodeMatch, l)) saw all-zero change stats
        # on the sampler's working copies
        rng = Random.Xoshiro(19)
        m = MultilayerNetwork(8; directed=true)
        add_layer!(m, :a)
        add_layer!(m, :b)
        set_vertex_attribute!(layer_network(m, :a), :group,
                              Dict(v => (v <= 4 ? "x" : "y") for v in 1:8))

        c = ERGMMulti._copy_net(layer_network(m, :a))
        @test get_vertex_attribute(c, :group, 5) == "y"

        terms = [LayerEdges(), WithinLayer(NodeMatch(:group), 1)]
        draws = simulate_multi_ergm(m, terms, [-2.0, 3.0];
                                    n_sim=20, burnin=3000, interval=200, rng=rng)
        @test get_vertex_attribute(layer_network(draws[1], :a), :group, 1) == "x"

        # Planted homophily expressed in layer 1: same-group ties outnumber
        # cross-group ones despite fewer same-group dyads (24 vs 32)
        same = 0
        cross = 0
        for d in draws
            for e in edges(layer_network(d, 1))
                ((src(e) <= 4) == (dst(e) <= 4)) ? (same += 1) : (cross += 1)
            end
        end
        @test same > cross

        # And fitting a draw recovers the homophily sign
        r = ergm_multi(draws[end], terms)
        @test r.converged
        @test r.coefficients[2] > 0
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

        # A one-line summary, not the default struct dump
        @test sprint(show, ml) == "MultilevelNetwork: 2 levels (level 1: 5 nodes, " *
                                  "3 edges; level 2: 2 nodes, 0 edges), 1 cross-level edge"
        @test !occursin("Dict", sprint(show, ml))
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
        # A one-line summary, not the default struct dump
        @test sprint(show, mn) == "MultiNetwork: 2 networks — x (3 vertices, 1 edges, " *
                                  "directed); y (4 vertices, 2 edges, directed)"
        @test !occursin("Network{Int64}[", sprint(show, mn))
    end

    @testset "StatsAPI accessors" begin
        m = fixture()
        r = ergm_multi(m, [LayerEdges(1), LayerEdges(2)])

        @test coef(r) == r.coefficients
        @test stderror(r) == r.std_errors
        V = vcov(r)
        @test size(V) == (2, 2)
        @test sqrt(V[1, 1]) ≈ r.std_errors[1]
        @test loglikelihood(r) == r.loglik
        @test aic(r) == r.aic
        @test bic(r) == r.bic
        @test nobs(r) == 24  # 2 layers × 12 ordered dyads
        @test dof(r) == 2

        # Offsets: fixed coefficients get NaN vcov rows and don't add dof
        ro = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                        offsets=Dict(1 => -1.0))
        @test isnan(vcov(ro)[1, 1])
        @test !isnan(vcov(ro)[2, 2])
        @test dof(ro) == 1
    end

    @testset "Dyad-dependence classification and caveat in show()" begin
        # Trait values (ERGM.is_dyad_dependent extended for multilayer terms;
        # in the multilayer model the dyads are (layer, i, j) triples, so
        # cross-layer terms are dyad-dependent too)
        @test !is_dyad_dependent(LayerEdges())
        @test !is_dyad_dependent(LayerEdges(1))
        @test !is_dyad_dependent(WithinLayer(Edges(), 1))
        @test !is_dyad_dependent(WithinLayer(NodeMatch(:g), 1))
        @test is_dyad_dependent(WithinLayer(Triangle(), 1))
        @test is_dyad_dependent(LayerMutual())
        @test is_dyad_dependent(LayerTriangle(2))
        @test is_dyad_dependent(InterlayerDependence(1, 2))
        @test is_dyad_dependent(MultiplexMutual(1, 2))

        m = fixture()

        # Dyad-independent formula → no caveat; renders through the shared
        # Networks.jl coefficient printer (R-style columns, signif codes)
        r_ind = ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
        out_ind = sprint(show, r_ind)
        @test !occursin("dyad-dependent", out_ind)
        @test occursin("Pr(>|z|)", out_ind)
        @test occursin("Std.Error", out_ind)
        @test count("Signif. codes:", out_ind) == 1

        # Dyad-dependent formula → pseudo-likelihood warning
        r_dep = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)])
        out = sprint(show, r_dep)
        @test occursin("dyad-dependent", out)
        @test occursin("pseudolikelihood", out)
        @test occursin("anticonservative", out)
    end

    @testset "Goodness of fit" begin
        m = fixture()
        r = ergm_multi(m, [LayerEdges(1), LayerEdges(2)])

        g = gof(r; n_sim=30, burnin=300, interval=30, rng=Random.Xoshiro(41))

        # gof extends Networks.jl's shared generic and returns the shared
        # GOFResult container
        @test ERGMMulti.gof === Networks.gof
        @test g isa GOFResult
        @test n_simulations(g) == 30
        @test length(g.statistics) == 2

        # Panel 1: the fitted terms' statistics
        stats = g.statistics[1]
        @test stats.name == "model statistics"
        @test stats.labels == ["L.edges.1", "L.edges.2"]
        @test stats.observed == [4.0, 4.0]
        @test all(0.0 .< stats.p_values .<= 1.0)
        # The saturated per-layer edges model should fit its own data
        @test all(stats.p_values .> 0.01)

        # Panel 2: per-layer edge counts (same observations here)
        le = g.statistics[2]
        @test le.name == "layer edges"
        @test le.labels == ["friendship", "advice"]
        @test le.observed == [4.0, 4.0]

        # Reproducible under the same seed
        g2 = gof(r; n_sim=30, burnin=300, interval=30, rng=Random.Xoshiro(41))
        @test g2.statistics[1].simulated == stats.simulated

        # ... and renders through the shared formatted display
        out = sprint(show, g)
        @test occursin("Goodness-of-fit assessment: Multilayer ERGM", out)
        @test occursin("MC p-value", out)
    end

    @testset "Entry points: fit_ergm_multi is canonical, fit_multi_ergm deprecated" begin
        # Item 16: the harmonised `fit_<model>` name is a `const` alias of the
        # R name, and the pre-0.2 `fit_multi_ergm` is a deprecated wrapper
        @test fit_ergm_multi === ergm_multi
        @test fit_multi_ergm !== ergm_multi
        m = fixture()
        r = ergm_multi(m, [LayerEdges(1)])
        r_dep = @test_deprecated fit_multi_ergm(m, [LayerEdges(1)])
        @test r_dep isa MultiERGMResult
        @test coef(r_dep) == coef(r)
        @test stderror(r_dep) == stderror(r)
        @test coef(fit_ergm_multi(m, [LayerEdges(1)])) == coef(r)
        # Keyword vocabulary: maxiter / rng / n_sim
        @test hasmethod(ergm_multi, Tuple{MultilayerNetwork, Any})
        kws = Base.kwarg_decl(only(methods(ergm_multi, Tuple{MultiERGMModel})))
        @test :maxiter in kws && :rng in kws && :se in kws
        skws = Base.kwarg_decl(only(methods(simulate_multi_ergm,
                                            Tuple{MultiERGMModel, AbstractVector{<:Real}})))
        @test :n_sim in skws && :rng in skws && :burnin in skws && :interval in skws
        # The estimator offers no `missing=` keyword: the only policy is :error
        @test missing_policies(ergm_multi) == (:error,)
        @test missing_policies(fit_ergm_multi) == (:error,)
    end

    @testset "Missing dyads are rejected" begin
        # MPLE enumerates every within-layer dyad as observed; a masked dyad
        # would enter the pseudo-likelihood at its face value.
        m = MultilayerNetwork(5; directed=false)
        add_layer!(m, :friendship)
        add_layer!(m, :advice)
        add_layer_edge!(m, 1, 1, 2)
        add_layer_edge!(m, 2, 2, 3)

        @test ergm_multi(m, [LayerEdges(1)]) isa MultiERGMResult

        # Absent-face masked dyad in the SECOND layer
        set_missing_dyad!(m.layers[2], 4, 5)
        @test_throws ArgumentError ergm_multi(m, [LayerEdges(1)])

        # The error names the offending layer
        msg = try
            ergm_multi(m, [LayerEdges(1)])
        catch e
            sprint(showerror, e)
        end
        @test occursin("layer 2", msg)

        # Present-face masked dyad is rejected just the same
        clear_missing_dyads!(m.layers[2])
        set_missing_dyad!(m.layers[2], 2, 3)
        @test_throws ArgumentError ergm_multi(m, [LayerEdges(1)])

        clear_missing_dyads!(m.layers[2])
        @test ergm_multi(m, [LayerEdges(1)]) isa MultiERGMResult
    end

    @testset "Result metadata protocol" begin
        m = fixture()

        # Same estimator (within-layer MPLE), two formulas: the edges-only model
        # is dyad-independent, so its pseudo-likelihood IS the likelihood.
        indep = ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
        md = fit_metadata(indep)
        @test md.estimand == :multilayer_ergm
        @test md.objective == :pseudolikelihood
        @test md.is_exact
        @test md.se_method == :hessian
        @test md.missing_method == :rejected
        @test isempty(md.approximations)

        # A cross-layer dependence term makes the same estimator approximate
        dep = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)])
        @test objective(dep) == :pseudolikelihood
        @test !is_exact(dep)
        @test any(occursin("anticonservative", a) for a in approximations(dep))

        # `show`'s prose caveat and the protocol are driven by one predicate
        @test occursin("pseudolikelihood", sprint(show, dep))
        @test !occursin("Warning", sprint(show, indep))

        # Offsets are fixed, not estimated — the protocol says so
        off = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                         offsets=Dict(1 => -1.0))
        @test any(occursin("offset", a) for a in approximations(off))
    end
    @testset "Robust standard errors: se=:bootstrap" begin
        # Issue #9 / ERGMMulti#1: the within-layer MPLE reported inverse-
        # pseudo-Hessian SEs with no robust alternative, and they are
        # anticonservative whenever a term is dyad-dependent over the (layer, i, j)
        # dyad universe. `se=:bootstrap` adds a parametric bootstrap (simulate at
        # θ̂ with `simulate_multi_ergm`, refit, empirical covariance) on the ONE
        # shared `Networks.bootstrap_cov` loop, with the same API as `ERGM.mple`.
        rng0 = MersenneTwister(2)
        n = 10
        l1, l2 = network(n; directed=true), network(n; directed=true)
        for i in 1:n, j in 1:n
            i == j && continue
            rand(rng0) < 0.25 && add_edge!(l1, i, j)
            rand(rng0) < 0.25 && add_edge!(l2, i, j)
        end
        m = as_multilayer([l1, l2], [:a, :b])
        terms = [LayerEdges(Colon()), LayerMutual(Colon())]   # dyad-DEPENDENT

        hess = ergm_multi(m, terms)
        boot = ergm_multi(m, terms; se=:bootstrap, n_boot=60,
                          rng=MersenneTwister(11))

        # The bootstrap replaces the COVARIANCE, not the point estimate
        @test coef(boot) == coef(hess)
        @test loglikelihood(boot) == loglikelihood(hess)
        @test aic(boot) == aic(hess)
        @test stderror(boot) != stderror(hess)
        @test vcov(boot) != vcov(hess)
        @test all(isfinite, stderror(boot))

        # Reproducible under a fixed rng
        boot2 = ergm_multi(m, terms; se=:bootstrap, n_boot=60,
                           rng=MersenneTwister(11))
        @test stderror(boot2) == stderror(boot)
        @test vcov(boot2) == vcov(boot)
        @test stderror(ergm_multi(m, terms; se=:bootstrap, n_boot=60,
                                  rng=MersenneTwister(12))) != stderror(boot)

        # The dependent term's robust SE EXCEEDS the Hessian one — that gap IS
        # the anticonservatism the issue is about. (The edges coefficient is
        # dyad-independent; its two SEs agree to sampling error, and the earlier
        # `all(boot .> hess)` held only because refits on boundary replicates
        # used to "converge" on the flat asymptote and inflate every column.)
        @test stderror(boot)[2] > stderror(hess)[2]
        @test 0.5 < stderror(boot)[1] / stderror(hess)[1] < 2.0

        # `se_method` reports what was ACTUALLY used, in both directions
        @test se_method(hess) === :hessian
        @test se_method(boot) === :bootstrap
        @test fit_metadata(hess).se_method === :hessian
        @test fit_metadata(boot).se_method === :bootstrap

        # ... and so does the printed output: the anticonservatism caveat is a
        # claim about the inverse Hessian, so it must NOT be made of a bootstrap
        out_h = sprint(show, hess)
        out_b = sprint(show, boot)
        @test occursin("inverse pseudo-Hessian", out_h)
        @test occursin("anticonservative", out_h)
        @test occursin("parametric bootstrap", out_b)
        @test !occursin("anticonservative", out_b)
        # The POINT ESTIMATE is an MPLE either way, and the bootstrap fit says so
        @test occursin("biased in", out_b)

        # The approximations list agrees with the printed prose (one predicate)
        @test any(occursin("anticonservative", a) for a in approximations(hess))
        @test !any(occursin("anticonservative", a) for a in approximations(boot))
        @test any(occursin("parametric bootstrap", a) for a in approximations(boot))

        # Offsets stay fixed under the bootstrap: a coefficient that was not
        # estimated carries no uncertainty, whatever the covariance estimator
        off = ergm_multi(m, terms; offsets=Dict(1 => -1.5), se=:bootstrap,
                         n_boot=20, rng=MersenneTwister(13))
        @test coef(off)[1] == -1.5
        @test isnan(stderror(off)[1])
        @test all(isnan, vcov(off)[1, :])
        @test isfinite(stderror(off)[2])
        @test se_method(off) === :bootstrap
        @test dof(off) == 1

        # A dyad-INDEPENDENT model needs no caveat under either estimator: there
        # the pseudo-likelihood IS the likelihood
        indep = ergm_multi(m, [LayerEdges(Colon())])
        @test is_exact(indep)
        @test !occursin("anticonservative", sprint(show, indep))

        # Unknown se symbols are rejected, not silently ignored
        @test_throws ArgumentError ergm_multi(m, terms; se=:sandwich)
        @test_throws ArgumentError ergm_multi(m, terms; se=:bootstrap, n_boot=1)
    end

    @testset "MultilayerNetwork{D}: directedness is a type parameter (item 19)" begin
        m = fixture()
        @test m isa MultilayerNetwork{true}
        @test is_directed(m)
        @test is_directed(typeof(m))
        @test isconcretetype(fieldtype(typeof(m), :layers))
        @test eltype(m.layers) === Network{Int, true}
        @test !hasfield(typeof(m), :directed)
        mu = MultilayerNetwork(4; directed=false)
        @test mu isa MultilayerNetwork{false}
        @test !is_directed(mu)
        @test eltype(mu.layers) === Network{Int, false}
        @test_throws ArgumentError MultilayerNetwork{1}(3)
        @test_throws ArgumentError MultilayerNetwork(-1)

        # A layer of the other directedness is refused with a message naming both
        msg = errmsg(() -> add_layer!(m, :x; net=network(4; directed=false)))
        @test occursin("undirected", msg) && occursin("directed", msg) &&
              occursin("MultilayerNetwork{true}", msg)
        msg_u = errmsg(() -> add_layer!(mu, :x; net=network(4; directed=true)))
        @test occursin(":x is directed", msg_u) && occursin("must be undirected", msg_u)
        @test occursin("4 vertices", errmsg(() -> add_layer!(m, :y; net=network(3))))

        # Conversions construct the typed form
        @test as_multilayer([network(3; directed=false)], [:a]) isa MultilayerNetwork{false}
        @test split_by_layer(combine_networks(m), 4, 2) isa MultilayerNetwork{true}
        c = combine_networks(fixture_undirected())
        @test !is_directed(c)
        @test split_by_layer(c, 5, 2) isa MultilayerNetwork{false}
        @test occursin("undirected", sprint(show, mu))

        # The model and result types carry the parameter
        model = MultiERGMModel([LayerEdges(1)], m)
        @test model isa MultiERGMModel{true}
        @test is_directed(model)
        @test isconcretetype(fieldtype(typeof(model), :network))
        fit = ergm_multi(model)
        @test fit isa MultiERGMResult{true}
        @test is_directed(fit)
        @test ergm_multi(fixture_undirected(), [LayerEdges()]) isa MultiERGMResult{false}
    end

    @testset "Model construction validates the formula (item 12 inheritance)" begin
        m = fixture()
        mu = fixture_undirected()

        # (i) Out-of-range layers throw at construction, naming the layer and
        # the network's layers — they used to yield all-zero coefficients,
        # NaN standard errors and converged = false
        for bad in (LayerEdges(3), LayerMutual([1, 3]), LayerTriangle(0),
                    WithinLayer(Edges(), 3), InterlayerDependence(1, 3),
                    MultiplexMutual(3, 2))
            @test_throws ArgumentError MultiERGMModel([bad], m)
            @test_throws ArgumentError ergm_multi(m, [bad])
            msg = errmsg(() -> ergm_multi(m, [bad]))
            @test occursin("layer", msg) && occursin(":friendship", msg) &&
                  occursin(":advice", msg) && occursin("1:2", msg)
        end
        @test occursin("layer 3", errmsg(() -> ergm_multi(m, [LayerEdges(3)])))
        @test_throws ArgumentError MultiERGMModel([LayerEdges(Int[])], m)
        @test_throws ArgumentError simulate_multi_ergm(m, [LayerEdges(3)], [0.0]; n_sim=1)

        # A network with no within-layer dyads (n ≤ 1) is refused at
        # construction: it used to run Newton on an empty design and return
        # coef = [0.0], NaN standard errors, nobs = 0 and two misleading
        # non-convergence warnings ("Hessian not negative definite", "raise
        # maxiter") — there is simply nothing to estimate
        for n0 in (0, 1), directed in (true, false)
            m0 = MultilayerNetwork(n0; directed=directed)
            add_layer!(m0, :a)
            @test_throws ArgumentError MultiERGMModel([LayerEdges()], m0)
            @test_throws ArgumentError ergm_multi(m0, [LayerEdges()])
            @test_throws ArgumentError simulate_multi_ergm(m0, [LayerEdges()], [0.0])
            msg = errmsg(() -> ergm_multi(m0, [LayerEdges()]))
            @test occursin("no within-layer dyads", msg) && occursin("n = $n0 actor", msg) &&
                  occursin("at least two actors", msg)
        end
        # ... and a 2-actor network is the smallest one that fits
        m2 = MultilayerNetwork(2; directed=true); add_layer!(m2, :a); add_layer_edge!(m2, :a, 1, 2)
        @test coef(ergm_multi(m2, [LayerEdges()])) ≈ [0.0]     # logit(1/2)

        # (ii) WithinLayer terms are validated against their layer with
        # ERGM.jl's own validator: a missing attribute is named ...
        msg = errmsg(() -> ergm_multi(m, [LayerEdges(), WithinLayer(NodeMatch(:welth), 1)]))
        @test occursin(":welth", msg) && occursin("does not exist", msg)
        # ... a partial attribute is refused (statnet refuses NA) ...
        mp = fixture()
        set_vertex_attribute!(mp.layers[1], :grp, Dict(1 => "a", 2 => "a", 3 => "b"))
        msg = errmsg(() -> ergm_multi(mp, [LayerEdges(), WithinLayer(NodeMatch(:grp), 1)]))
        @test occursin(":grp", msg) && occursin("every vertex", msg) && occursin("4", msg)
        # ... but the same attribute on the OTHER layer is not required
        set_vertex_attribute!(mp.layers[1], :grp, 4, "b")
        @test ergm_multi(mp, [LayerEdges(), WithinLayer(NodeMatch(:grp), 1)]) isa MultiERGMResult
        @test_throws ArgumentError ergm_multi(mp, [LayerEdges(), WithinLayer(NodeMatch(:grp), 2)])
        # ... an EdgeCov of the wrong size ...
        msg = errmsg(() -> ergm_multi(m, [LayerEdges(), WithinLayer(EdgeCov(zeros(3, 3); name="edgecov.w"), 1)]))
        @test occursin("3×3", msg) && occursin("4 vertices", msg)
        # ... and direction requirements, in both directions
        msg = errmsg(() -> ergm_multi(m, [WithinLayer(Degree(1), 1)]))
        @test occursin("undirected networks", msg) && occursin("directed", msg)
        msg = errmsg(() -> ergm_multi(mu, [WithinLayer(Mutual(), 1)]))
        @test occursin("only defined for directed networks", msg)
        @test_throws ArgumentError ergm_multi(mu, [WithinLayer(OStar(2), 2)])

        # (iii) LayerMutual / MultiplexMutual are directed-only and are refused
        # on undirected data with ERGM's sentence (they used to return 0.0)
        @test requires_directed(LayerMutual())
        @test requires_directed(MultiplexMutual(1, 2))
        @test !requires_directed(LayerEdges())
        @test requires_undirected(WithinLayer(Kstar(2), 1))
        @test requires_directed(WithinLayer(Mutual(), 1))
        for bad in (LayerMutual(), LayerMutual(1), MultiplexMutual(1, 2))
            msg = errmsg(() -> ergm_multi(mu, [LayerEdges(), bad]))
            @test occursin("only defined for directed networks", msg) &&
                  occursin(name(bad), msg)
            @test_throws ArgumentError compute(bad, mu)
            @test_throws ArgumentError change_stat_layer(bad, mu, 1, 1, 2)
        end
        @test compute(LayerMutual(), m) == 2.0     # still fine on directed data

        # (iv) offsets index terms and leave something to estimate
        @test_throws ArgumentError MultiERGMModel([LayerEdges()], m; offsets=Dict(2 => 0.0))
        @test_throws ArgumentError MultiERGMModel([LayerEdges()], m; offsets=Dict(1 => 0.0))
        @test_throws ArgumentError MultiERGMModel([LayerEdges()], m; offsets=Dict(1 => NaN, 2 => 0.0))
        @test occursin("nothing to estimate",
                       errmsg(() -> ergm_multi(m, [LayerEdges(1), LayerEdges(2)];
                                               offsets=Dict(1 => 0.0, 2 => 0.0))))
        # A term list that is not a term list
        @test_throws ArgumentError MultiERGMModel([LayerEdges], m)
        @test_throws ArgumentError MultiERGMModel(AbstractERGMTerm[], m)
        @test occursin("term type", errmsg(() -> MultiERGMModel([LayerEdges], m)))
        @test_throws ArgumentError MultiERGMModel([LayerEdges()], MultilayerNetwork(4))

        # (v) A bare ERGM.jl term — `Mutual()` where `WithinLayer(Mutual(), l)`
        # was meant, the ergm.multi migrant's `L(~mutual, ~A)` slip — is refused
        # by the constructor with the fix spelled out; it used to construct and
        # die inside the design builder with a MethodError on
        # `change_stat_layer(::Mutual, ...)`
        for bare in (Mutual(), Edges(), Triangle(), NodeMatch(:grp))
            @test_throws ArgumentError MultiERGMModel([LayerEdges(), bare], m)
            @test_throws ArgumentError ergm_multi(m, [LayerEdges(), bare])
            @test_throws ArgumentError simulate_multi_ergm(m, [bare], [0.0]; n_sim=1)
        end
        msg = errmsg(() -> ergm_multi(m, [LayerEdges(), Mutual()]))
        @test occursin("term 'mutual' (Mutual)", msg) &&
              occursin("WithinLayer(Mutual(...), l)", msg) &&
              occursin("L(~mutual, ~A)", msg) && !occursin("MethodError", msg)
        @test ergm_multi(m, [LayerEdges(), WithinLayer(Mutual(), 1)]) isa MultiERGMResult
        # ... and so is a multilevel descriptive, which never enters a fit
        msg = errmsg(() -> MultiERGMModel([LayerEdges(), Nestedness(1)], m))
        @test occursin("multilevel", msg) && occursin("never enter a fit", msg)
        @test !ERGMMulti._has_multilayer_change_stat(Mutual())
        @test ERGMMulti._has_multilayer_change_stat(WithinLayer(Mutual(), 1))
        @test ERGMMulti._has_multilayer_change_stat(InterlayerDependence(1, 2))

        # (vi) A layer selected by NAME inside a term is an ArgumentError with
        # the migration note (every other verb takes names; terms take
        # indices) — not a MethodError with a 'Closest candidates' dump
        for f in (() -> LayerEdges(:friendship), () -> LayerMutual(:advice),
                  () -> LayerTriangle([:friendship, :advice]),
                  () -> WithinLayer(Triangle(), :friendship),
                  () -> InterlayerDependence(:friendship, :advice),
                  () -> InterlayerDependence(1, :advice),
                  () -> MultiplexMutual(:friendship, 2))
            @test_throws ArgumentError f()
            msg = errmsg(f)
            @test occursin("by index, not by name", msg) &&
                  occursin("findfirst(==(", msg) && occursin("layer_names(m)", msg)
        end
        @test occursin("use LayerEdges(k) with k = findfirst(==(:friendship), layer_names(m))",
                       errmsg(() -> LayerEdges(:friendship)))
        @test occursin("use WithinLayer(term, k)", errmsg(() -> WithinLayer(Triangle(), :advice)))
        @test occursin("use InterlayerDependence(k1, k2)",
                       errmsg(() -> InterlayerDependence(:friendship, :advice)))
        @test occursin("got [:friendship, :advice]",
                       errmsg(() -> LayerTriangle([:friendship, :advice])))
        # The index the message points to works
        k = findfirst(==(:advice), layer_names(m))
        @test compute(LayerEdges(k), m) == compute(LayerEdges(2), m)

        # (vii) A layer that CONTAINS a self-loop is refused by the model
        # constructor (as `ERGM.ERGMModel` does): the statistics would count
        # it, but the within-layer MPLE design, nobs, the sampler and gof
        # range over the off-diagonal dyads only. A `loops=true` layer with no
        # loop is accepted.
        la = network(5; directed=true, loops=true)
        add_edge!(la, 1, 2); add_edge!(la, 2, 1); add_edge!(la, 1, 1)
        lb = network(5; directed=true, loops=true)
        add_edge!(lb, 3, 4)
        mloop = as_multilayer([lb, la], [:a, :b])              # accepted by the adapter
        @test compute(LayerEdges(2), mloop) == 3.0             # the statistic counts the loop
        @test_throws ArgumentError MultiERGMModel([LayerEdges()], mloop)
        @test_throws ArgumentError ergm_multi(mloop, [LayerEdges(1), LayerEdges(2)])
        @test_throws ArgumentError simulate_multi_ergm(mloop, [LayerEdges()], [-1.0]; n_sim=1)
        msg = errmsg(() -> ergm_multi(mloop, [LayerEdges()]))
        @test occursin("layer :b (layer 2) contains 1 self-loop (at vertex 1)", msg) &&
              occursin("This network contains loops", msg) &&
              occursin("rem_edge!(layer_network(m, 2), v, v)", msg)
        rem_edge!(la, 1, 1)
        fl = ergm_multi(mloop, [LayerEdges(1), LayerEdges(2)])   # loops=true, no loop: fine
        @test coef(fl) ≈ [log(1 / 19), log(2 / 18)] atol = 1e-8
        @test nobs(fl) == 40

        # Both entry points go through the constructor; a model can be reused
        model = MultiERGMModel([LayerEdges(1), LayerEdges(2)], m)
        @test coef(ergm_multi(model)) == coef(ergm_multi(m, [LayerEdges(1), LayerEdges(2)]))
        # ... and prints a one-line summary consistent with `ERGM.ERGMModel`
        @test sprint(show, model) == "MultiERGMModel{true}: 4 actors, 2 directed layers " *
                                     "(:friendship, :advice); terms: L.edges.1 + L.edges.2"
        @test sprint(show, MultiERGMModel([LayerEdges(1), LayerEdges(2)], m;
                                          offsets=Dict(2 => -2.0))) ==
              "MultiERGMModel{true}: 4 actors, 2 directed layers (:friendship, :advice); " *
              "terms: L.edges.1 + L.edges.2; offsets: 2 => -2.0"
        @test startswith(sprint(show, MultiERGMModel([LayerEdges()], mu)),
                         "MultiERGMModel{false}: 5 actors, 2 undirected layers")
        @test !occursin("TermSet", sprint(show, model))
        @test simulate_multi_ergm(model, [-1.0, -1.0]; n_sim=2, burnin=10, interval=5,
                                  rng=Xoshiro(1)) isa Vector{MultilayerNetwork{true}}
        @test_throws ArgumentError simulate_multi_ergm(model, [-1.0]; n_sim=1)
    end

    @testset "Materialized within-layer terms and expanding terms" begin
        # An attribute term inside WithinLayer is snapshotted at construction
        # (dense vector, not a Dict lookup per toggle) and keeps its label
        rng = Xoshiro(7)
        m = MultilayerNetwork(8; directed=true)
        for l in 1:2
            net = network(8; directed=true)
            for v in 1:8
                set_vertex_attribute!(net, :grp, v, v <= 4 ? "x" : "y")
                set_vertex_attribute!(net, :w, v, Float64(v))
            end
            for i in 1:8, j in 1:8
                i != j && rand(rng) < 0.3 && add_edge!(net, i, j)
            end
            add_layer!(m, Symbol("L", l); net=net)
        end
        model = MultiERGMModel([LayerEdges(), WithinLayer(NodeMatch(:grp), 1),
                                WithinLayer(NodeCov(:w), 2)], m)
        @test model.terms.names == ["L.edges.all", "L1.nodematch.grp", "L2.nodecov.w"]
        @test model.terms[2] isa WithinLayer
        @test !(model.terms[2].term isa NodeMatch)          # materialized twin
        @test name(model.terms[2].term) == "nodematch.grp"
        @test compute_all(model.terms, m) ==
              [compute(LayerEdges(), m), compute(WithinLayer(NodeMatch(:grp), 1), m),
               compute(WithinLayer(NodeCov(:w), 2), m)]
        # ... and the materialized change statistics agree with the raw term's
        for i in 1:8, j in 1:8
            i == j && continue
            @test change_stat_layer(model.terms[2], m, 1, i, j) ==
                  change_stat_layer(WithinLayer(NodeMatch(:grp), 1), m, 1, i, j)
            @test change_stat_layer(model.terms[3], m, 2, i, j) ==
                  change_stat_layer(WithinLayer(NodeCov(:w), 2), m, 2, i, j)
        end

        # A multi-level NodeFactor / a Degree range expands into one statistic
        # per level / degree, exactly as in ERGM.jl; offsets follow the term
        # they were given for
        for v in 1:8
            set_vertex_attribute!(m.layers[1], :cls, v, ("a", "b", "c")[mod1(v, 3)])
        end
        mx = MultiERGMModel([WithinLayer(NodeFactor(:cls), 1), LayerEdges(2)], m;
                            offsets=Dict(2 => -1.0))
        @test mx.terms.names == ["L1.nodefactor.cls.b", "L1.nodefactor.cls.c", "L.edges.2"]
        @test mx.offsets == Dict(3 => -1.0)
        fx = ergm_multi(mx)
        @test coef(fx)[3] == -1.0 && isnan(stderror(fx)[3])
        @test coeftable(fx).names[3] == "L.edges.2 (offset)"
        # An offset on the expanding term itself is refused with the fix
        msg = errmsg(() -> MultiERGMModel([WithinLayer(NodeFactor(:cls), 1), LayerEdges(2)], m;
                                          offsets=Dict(1 => 0.5)))
        @test occursin("expands into 2 statistics", msg) && occursin("level=", msg)
        mu = fixture_undirected()
        md = MultiERGMModel([LayerEdges(), WithinLayer(Degree(0:1), 1)], mu)
        @test md.terms.names == ["L.edges.all", "L1.degree0", "L1.degree1"]
        @test length(simulate_multi_ergm(md, [-1.0, 0.0, 0.0]; n_sim=1, burnin=5, interval=1)) == 1
    end

    @testset "Direction-aware coefficient labels: name(term, m)" begin
        m = fixture()
        mu = fixture_undirected()
        @test name(WithinLayer(GWESP(0.5), 1), m) == "L1.gwesp.OTP.fixed.0.5"
        @test name(WithinLayer(GWESP(0.5), 1), mu) == "L1.gwesp.fixed.0.5"
        @test name(WithinLayer(GWDSP(0.5), 2), m) == "L2.gwdsp.OTP.fixed.0.5"
        @test name(WithinLayer(Edges(), 2), m) == "L2.edges"
        @test name(WithinLayer(GWESP(0.5), 1)) == "L1.gwesp.fixed.0.5"   # network-free form
        @test name(LayerEdges(1), m) == name(LayerEdges(1)) == "L.edges.1"
        @test_throws ArgumentError name(WithinLayer(Edges(), 3), m)
        # ... and the fitted table carries the same labels as ERGM.jl / R would
        # (the 4-actor fixture has no shared partner, so gwesp sits at its
        # boundary and the fit warns as R does — the labels are the point here)
        fit = @test_logs (:warn, r"attainable") match_mode=:any ergm_multi(
            m, [LayerEdges(1), WithinLayer(GWESP(0.5), 1)])
        @test coeftable(fit).names == ["L.edges.1", "L1.gwesp.OTP.fixed.0.5"]
        @test fit.model.terms.names == coeftable(fit).names
        fit_u = ergm_multi(mu, [LayerEdges(1), WithinLayer(GWESP(0.5), 1)])
        @test coeftable(fit_u).names == ["L.edges.1", "L1.gwesp.fixed.0.5"]
        g = gof(fit_u; n_sim=5, burnin=20, interval=5, rng=Xoshiro(2))
        @test g.statistics[1].labels == ["L.edges.1", "L1.gwesp.fixed.0.5"]
        ef = @test_logs (:warn, r"attainable") match_mode=:any ergm_multi(
            m, [LayerEdges(1), WithinLayer(GWESP(0.5), 1)])
        @test sprint(show, ef) == sprint(show, fit)
    end

    @testset "StatsAPI surface is complete (item 15)" begin
        m = fixture()
        hess = ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
        boot = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                          se=:bootstrap, n_boot=12, boot_burnin=100, boot_interval=20,
                          rng=Xoshiro(5))
        off = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)];
                         offsets=Dict(1 => -1.0))
        for fit in (hess, boot, off)
            @test check_statsapi(fit; strict=true) !== nothing
            @test all(check_statsapi(fit))
            ci = confint(fit)
            @test size(ci) == (length(coef(fit)), 2)
            k = findall(!isnan, stderror(fit))
            @test all(ci[k, 1] .< coef(fit)[k] .< ci[k, 2])
            ci90 = confint(fit; level=0.9)
            @test all(ci90[k, 2] .- ci90[k, 1] .< ci[k, 2] .- ci[k, 1])
            tbl = coeftable(fit)
            @test tbl isa CoefficientTable
            @test tbl.estimates == coef(fit)
            @test isequal(tbl.std_errors, stderror(fit))          # NaN-aware
            @test isequal(tbl.p_values, z_pvalues(coef(fit) ./ stderror(fit)))
            # The printed table IS the inspected one
            @test occursin(sprint(show, tbl), sprint(show, fit))
        end
        @test_throws ArgumentError confint(hess; level=1.5)
        # Offset rows: tagged, NaN interval, NaN p-value, excluded from dof
        @test coeftable(off).names == ["L.edges.all (offset)", "duplex.1.2"]
        @test all(isnan, confint(off)[1, :])
        @test isnan(coeftable(off)[1].p_value) && isnan(coeftable(off)[1].z_value)
        @test !isnan(coeftable(off)["duplex.1.2"].p_value)
        @test dof(off) == 1
        @test occursin("(offset)", sprint(show, off))
        # z → p is the shared Networks helper (floored, NaN-aware)
        @test coeftable(hess).p_values == z_pvalues(coef(hess), stderror(hess)).p
    end

    @testset "Cross-package reach-ins target public bindings (item 13)" begin
        src = read(joinpath(@__DIR__, "..", "src", "ERGMMulti.jl"), String)
        reach = Set{Tuple{Symbol, Symbol}}()
        for mod in (:ERGM, :Networks)
            for mt in eachmatch(Regex("\\b$(mod)\\.(_\\w+)"), src)
                push!(reach, (mod, Symbol(mt.captures[1])))
            end
        end
        @test !isempty(reach)     # the test must actually see the reach-ins
        # R's boundary-statistic drop, the separation test and their two
        # sentences are ERGM.jl's `public` helpers (held as `@test_broken`
        # until the 2026-09 reconciliation declared them). The boundary
        # sentence is emitted through `ERGM._warn_boundary(...; note=)`, whose
        # default parenthesis "R ergm reports the same" is false of ergm.multi
        # — `_warn_multi_boundary` replaces it, never restates the sentence.
        for (mod, nm) in reach
            M = mod === :ERGM ? ERGM : Networks
            @test isdefined(M, nm)
            @test Base.ispublic(M, nm)
        end
        for nm in (:_boundary_columns_iterated, :_separated, :_warn_separated, :_warn_boundary)
            @test (:ERGM, nm) in reach
        end
        # No private z→p / Newton / dyad-dependence copies: the shared bindings
        @test !isdefined(ERGMMulti, :_has_dyad_dependent)
        @test !isdefined(ERGMMulti, :_z_pvalues)
        @test ERGMMulti.has_dyad_dependent === ERGM.has_dyad_dependent
        @test ERGMMulti.newton_fit === Networks.newton_fit
        @test ERGMMulti.logistic_derivatives === Networks.logistic_derivatives
        @test ERGMMulti.z_pvalues === Networks.z_pvalues
        @test ERGMMulti.check_se === Networks.check_se
        @test ERGMMulti.mh_toggle! === ERGM.mh_toggle!
        @test !occursin("ERGM._z_pvalues", src)
        @test !occursin("ERGM.newton_fit", src) && !occursin("import ERGM: newton_fit", src)
        # `has_dyad_dependent` is the one predicate behind show/is_exact
        m = fixture()
        @test !has_dyad_dependent(MultiERGMModel([LayerEdges(1), LayerEdges(2)], m))
        @test has_dyad_dependent(MultiERGMModel([LayerEdges(), LayerMutual()], m))
        @test has_dyad_dependent(MultiERGMModel([WithinLayer(Triangle(), 1)], m))
        # The shared `se=` validator: context and both allowed symbols
        msg = errmsg(() -> ergm_multi(m, [LayerEdges()]; se=:sandwich))
        @test occursin("ergm_multi", msg) && occursin(":hessian", msg) &&
              occursin(":bootstrap", msg) && occursin(":sandwich", msg)
    end

    @testset "Sampler runs on the shared Metropolis kernel (item 28)" begin
        m = fixture()
        terms = [LayerEdges(), InterlayerDependence(1, 2)]
        θ = [-1.0, 1.5]
        # Seeded reproducibility and rng independence from Threads
        a = simulate_multi_ergm(m, terms, θ; n_sim=5, burnin=200, interval=20, rng=Xoshiro(9))
        b = simulate_multi_ergm(m, terms, θ; n_sim=5, burnin=200, interval=20, rng=Xoshiro(9))
        @test [compute(LayerEdges(), s) for s in a] == [compute(LayerEdges(), s) for s in b]
        @test all(a[k].layers[l].graph == b[k].layers[l].graph for k in 1:5, l in 1:2)
        c = simulate_multi_ergm(m, terms, θ; n_sim=5, burnin=200, interval=20, rng=Xoshiro(10))
        @test any(a[k].layers[l].graph != c[k].layers[l].graph for k in 1:5, l in 1:2)
        # The kernel is exactly ERGM.mh_toggle! with (layer, i, j) moves: the
        # same rng stream through a hand-written call of the kernel gives the
        # same draws
        model = MultiERGMModel(terms, m)
        L, n = 2, 4
        cur = as_multilayer([copy(net) for net in m.layers], layer_names(m))
        out = MultilayerNetwork{true}[]
        delta = zeros(2)
        ERGM.mh_toggle!(Xoshiro(9), θ, delta,
                        rng -> begin
                            l = rand(rng, 1:L); i = rand(rng, 1:n)
                            j = rand(rng, 1:(n - 1)); j >= i && (j += 1)
                            (l, i, j)
                        end,
                        (d, mv) -> (ERGMMulti._change_stat_layer_all!(d, model.terms, cur, mv...);
                                    has_edge(cur.layers[mv[1]], mv[2], mv[3])),
                        (mv, rem) -> (rem ? rem_edge!(cur.layers[mv[1]], mv[2], mv[3]) :
                                            add_edge!(cur.layers[mv[1]], mv[2], mv[3]); nothing),
                        k -> push!(out, as_multilayer([copy(x) for x in cur.layers],
                                                      layer_names(m)));
                        burnin=200, interval=20, n_samples=5)
        @test all(a[k].layers[l].graph == out[k].layers[l].graph for k in 1:5, l in 1:2)

        # Undirected layers: only i < j dyads are ever toggled, every one of them
        mu = fixture_undirected()
        du = simulate_multi_ergm(mu, [LayerEdges()], [0.5]; n_sim=20, burnin=500,
                                 interval=20, rng=Xoshiro(3))
        @test all(!is_directed(s) for s in du)
        @test all(s isa MultilayerNetwork{false} for s in du)
        touched = Set{Tuple{Int, Int}}()
        for s in du, l in 1:2, e in edges(s.layers[l])
            push!(touched, (src(e), dst(e)))
        end
        @test length(touched) == 10        # all C(5,2) unordered dyads reached

        # Dyad-scaled defaults come from the ONE rule (ERGM._mcmc_defaults)
        d = ERGM._mcmc_defaults(ERGMMulti._n_within_dyads(m))
        @test ERGMMulti._resolve_multi_mcmc(m, nothing, nothing) == (d.burnin, d.interval)
        @test ERGMMulti._resolve_multi_mcmc(m, 7, nothing) == (7, d.interval)
        @test ERGMMulti._resolve_multi_mcmc(m, nothing, 3) == (d.burnin, 3)
        @test d.burnin == 20 * 24 && d.interval == 100
        @test_throws ArgumentError simulate_multi_ergm(m, terms, θ; n_sim=1, burnin=-1)
        @test_throws ArgumentError simulate_multi_ergm(m, terms, θ; n_sim=1, interval=0)
        @test simulate_multi_ergm(m, terms, θ; n_sim=0, burnin=1) == MultilayerNetwork{true}[]

        # No per-step allocation from the kernel + the generated change-stat
        # fill: a boxed value per step would cost ≥ 16 bytes/step. What remains
        # (measured ≈ 0.4 bytes/step) is the graph store growing its adjacency
        # vectors on accepted additions, not the sampler.
        function steps_alloc(steps)
            rng = Xoshiro(1)
            simulate_multi_ergm(model, θ; n_sim=0, burnin=steps, rng=rng)
            return @allocated simulate_multi_ergm(model, θ; n_sim=0, burnin=steps, rng=rng)
        end
        a1 = steps_alloc(2_000)
        a2 = steps_alloc(200_000)
        @test (a2 - a1) / 198_000 < 2.0
    end

    @testset "A boundary statistic is loud: R's drop semantics" begin
        # InterlayerDependence with NO co-occurring tie: its statistic sits at
        # the smallest attainable value, so the pseudo-likelihood has no finite
        # maximum. Newton used to "converge" on the flat asymptote (−21, SE
        # 19,000) and report converged = true. Now, as R ergm's `drop=TRUE`
        # for one-mode terms (ergm.multi 0.3.0 itself does NOT drop — see the
        # fixture assertions at the end of this testset): R's warning, the
        # coefficient fixed at -Inf with SE 0, the other coefficients fit on
        # the dyads the dropped statistic does not touch, and every honesty
        # channel says so.
        m = MultilayerNetwork(6; directed=true)
        add_layer!(m, :a); add_layer!(m, :b)
        for (i, j) in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]
            add_layer_edge!(m, :a, i, j)
            add_layer_edge!(m, :b, j, i)
        end
        @test compute(InterlayerDependence(1, 2), m) == 0.0
        fit = @test_logs (:warn, r"duplex.1.2 are at their smallest attainable values") match_mode=:any ergm_multi(
            m, [LayerEdges(), InterlayerDependence(1, 2)])
        # The sentence is R ergm's, but its closing parenthesis tells the
        # ergm.multi migrant the truth: ergm.multi 0.3.0 does NOT drop (it
        # warns "The MPLE does not exist!" and returns a finite value — the
        # fixture assertions below), so the warning must not say "R ergm
        # reports the same" (ERGM.jl's `_warn_boundary` wording, which is
        # true of `ergm` and false of `ergm.multi`)
        blogs, _ = Test.collect_test_logs() do
            ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)])
        end
        bmsg = only([string(l.message) for l in blogs if occursin("attainable", string(l.message))])
        @test startswith(bmsg, "ergm_multi: observed statistic(s) duplex.1.2 are at their smallest attainable values. Their coefficients will be fixed at -Inf")
        @test occursin("R ergm's drop=TRUE", bmsg) &&
              occursin("ergm.multi 0.3.0 warns \"The MPLE does not exist!\"", bmsg) &&
              occursin("returns a finite value", bmsg) && occursin("estimation guide", bmsg)
        @test !occursin("R ergm reports the same", bmsg)
        # ... one sentence, ERGM's, with the parenthesis passed as `note=`
        @test occursin("ERGM._warn_boundary(names, boundary; context=\"ergm_multi\",",
                       read(joinpath(@__DIR__, "..", "src", "ERGMMulti.jl"), String))
        @test occursin("exists; R ergm's drop=TRUE", bmsg)
        @test coef(fit)[2] == -Inf
        @test stderror(fit)[2] == 0.0
        @test coeftable(fit)[2].p_value == 0.0
        # 60 within-layer dyads, 12 touched by the dropped column (the 6 reversed
        # ties per layer), 12 ties among the 48 untouched: logit(12/48)
        @test coef(fit)[1] ≈ log(12 / 36) atol = 1e-8
        @test fit.converged
        @test !is_exact(fit)
        @test dof(fit) == 1
        @test bic(fit) ≈ -2 * loglikelihood(fit) + log(48) atol = 1e-10
        @test any(occursin("-Inf", a) && occursin("attainable", a) for a in approximations(fit))
        out = sprint(show, fit)
        @test occursin("-Inf", out) && occursin("attainable", out) && occursin("Note:", out)
        @test isequal(confint(fit)[2, :], [-Inf, -Inf])
        # The same design with the dependence term absent agrees on the edges
        # coefficient only if it also drops the touched rows — it does not, so
        # the two differ (the drop is a real refit, not a relabelling)
        plain = ergm_multi(m, [LayerEdges()])
        @test coef(plain)[1] ≈ log(12 / 48) atol = 1e-8
        @test coef(plain)[1] != coef(fit)[1]

        # Nothing can be SIMULATED at -Inf: `gof` and both `simulate_multi_ergm`
        # forms refuse the fit, naming the term. They used to run a frozen
        # chain (θ'Δ = -Inf·0 = NaN, every proposal rejected) and `gof` reported
        # p = 1.0 for every statistic on 20 copies of the observed network.
        @test_throws ArgumentError gof(fit; n_sim=5)
        msg = errmsg(() -> gof(fit; n_sim=5))
        @test occursin("gof: every coefficient must be finite", msg) &&
              occursin("duplex.1.2 = -Inf", msg) && occursin("drop=TRUE", msg)
        @test_throws ArgumentError simulate_multi_ergm(fit.model, coef(fit); n_sim=1)
        @test_throws ArgumentError simulate_multi_ergm(m, [LayerEdges(), InterlayerDependence(1, 2)],
                                                       [-2.1, -Inf]; n_sim=1)
        @test_throws ArgumentError simulate_multi_ergm(m, [LayerEdges(), InterlayerDependence(1, 2)],
                                                       [NaN, -2.5]; n_sim=1)
        msg = errmsg(() -> simulate_multi_ergm(m, [LayerEdges(), InterlayerDependence(1, 2)],
                                               [NaN, Inf]; n_sim=1))
        @test occursin("simulate_multi_ergm: every coefficient must be finite", msg) &&
              occursin("L.edges.all = NaN", msg) && occursin("duplex.1.2 = Inf", msg)
        # The remedy the message names — drop the term — simulates and assesses
        g_plain = gof(plain; n_sim=5, burnin=50, interval=10, rng=Xoshiro(3))
        @test g_plain isa GOFResult

        # R ergm.multi 0.3.0 on this very design (test/fixtures/twolayer_layer_terms.toml,
        # section iii): its layer operator does NOT propagate the attainable
        # range, so R warns "The MPLE does not exist!" and returns a FINITE
        # coefficient (~-17.6, SE ~2000) where ERGMMulti fixes -Inf; the finite
        # coefficients agree exactly. A documented divergence, pinned.
        g = load_golden(joinpath(@__DIR__, "fixtures", "twolayer_layer_terms.toml"))
        @test Int(g.values["boundary_n_actors"]) == 6
        @test Int.(g.values["boundary_layer_a_src"]) == [1, 2, 3, 4, 5, 6]
        @test Int.(g.values["boundary_layer_a_dst"]) == [2, 3, 4, 5, 6, 1]
        @test g.values["boundary_julia_terms"] == ["LayerEdges(1)", "LayerEdges(2)",
                                                   "InterlayerDependence(1, 2)"]
        bterms = [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2)]
        @test [compute(t, m) for t in bterms] == Float64.(g.values["boundary_statistics"]) == [6, 6, 0]
        @test g.values["boundary_mple_exists_warning"] == true
        r_coef = Float64.(g.values["boundary_r_coefficients"])
        @test all(isfinite, r_coef) && r_coef[3] < -15          # R: finite, not dropped
        @test Float64.(g.values["boundary_r_std_errors"])[3] > 1000
        bfit = @test_logs (:warn, r"duplex.1.2 are at their smallest attainable values") match_mode=:any ergm_multi(m, bterms)
        @test coef(bfit)[3] == -Inf && stderror(bfit)[3] == 0.0
        @test check_golden(g, "boundary_finite_coefficients", coef(bfit)[1:2]) ||
              error(golden_report(g, "boundary_finite_coefficients", coef(bfit)[1:2]))
        @test coef(bfit)[1:2] ≈ fill(log(6 / 18), 2) atol = 1e-10
        @test r_coef[1:2] ≈ coef(bfit)[1:2] atol = 1e-6

        # A bootstrap replicate at a boundary is excluded and reported
        mb = MultilayerNetwork(6; directed=true)
        add_layer!(mb, :a); add_layer!(mb, :b)
        for (i, j) in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (1, 3), (2, 4)]
            add_layer_edge!(mb, :a, i, j); add_layer_edge!(mb, :b, i, j)
        end
        add_layer_edge!(mb, :a, 3, 5)
        boot = ergm_multi(mb, [LayerEdges(), InterlayerDependence(1, 2)];
                          se=:bootstrap, n_boot=30, boot_burnin=100, boot_interval=20,
                          rng=Xoshiro(21))
        @test boot.boot_replicates isa Matrix{Float64}
        @test size(boot.boot_replicates) == (30, 2)
        n_bad = count(b -> !all(isfinite, view(boot.boot_replicates, b, :)), 1:30)
        if n_bad > 0
            @test any(occursin("excluded", a) for a in approximations(boot))
            @test occursin("excluded", sprint(show, boot))
        end
        @test all(isfinite, stderror(boot))

        # A genuinely unconverged Newton iteration is loud as well
        fit1 = @test_logs (:warn, r"did not converge") match_mode=:any ergm_multi(
            m, [LayerEdges(1), LayerEdges(2)]; maxiter=1)
        @test !fit1.converged
        @test !is_exact(fit1)
        @test any(occursin("did not", a) for a in approximations(fit1))
        out1 = sprint(show, fit1)
        @test occursin("converged: false", out1) && occursin("Warning: the Newton", out1)
        ok = ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
        @test ok.converged && is_exact(ok)
        @test !occursin("did not", sprint(show, ok))
    end


    @testset "Boundary statistics: README's single-edge design, separation, bootstrap refusal" begin
        # The README's pre-0.2 Quick Start: 30 actors, ONE friendship tie and
        # an empty advice layer, fit with pooled edges + cross-layer
        # dependence. No dyad is tied in both layers, so duplex.1.2 sits at
        # its smallest attainable value: R's drop, out loud.
        m = MultilayerNetwork(30; directed=true)
        add_layer!(m, :friendship); add_layer!(m, :advice)
        add_layer_edge!(m, :friendship, 1, 2)
        terms = [LayerEdges(), InterlayerDependence(1, 2)]
        fit = @test_logs (:warn, r"duplex.1.2 are at their smallest attainable values") match_mode=:any ergm_multi(m, terms)
        @test coef(fit)[2] == -Inf
        @test stderror(fit)[2] == 0.0
        @test coeftable(fit)[2].p_value == 0.0
        # 2 × 30 × 29 = 1740 within-layer dyads; the dropped column touches the
        # one advice dyad mirroring the friendship tie, leaving 1739 rows with
        # one tie: the pooled edges MLE is logit(1/1739) = log(1/1738)
        @test coef(fit)[1] ≈ log(1 / 1738) atol = 1e-8
        @test dof(fit) == 1
        @test bic(fit) ≈ -2 * loglikelihood(fit) + log(1739) atol = 1e-10
        @test fit.converged && !fit.separated
        @test !is_exact(fit)
        out = sprint(show, fit)
        @test occursin("duplex.1.2 fixed at -Inf", out) && occursin("smallest attainable", out)
        @test any(occursin("fixed at -Inf", a) for a in approximations(fit))
        # `se=:bootstrap` is refused: nothing can be simulated at -Inf
        msg = errmsg(() -> ergm_multi(m, terms; se=:bootstrap, n_boot=5))
        @test_throws ArgumentError ergm_multi(m, terms; se=:bootstrap, n_boot=5)
        @test occursin("±Inf", msg) && occursin("duplex.1.2", msg) && occursin("se=:hessian", msg)

        # A perfectly separated nodematch inside a layer: zero within-group
        # ties in layer 1 → -Inf, the layer-2 ties untouched
        ms = MultilayerNetwork(6; directed=false)
        add_layer!(ms, :a); add_layer!(ms, :b)
        for l in 1:2, v in 1:6
            set_vertex_attribute!(ms.layers[l], :grp, v, v <= 3 ? "x" : "y")
        end
        for (i, j) in [(1, 4), (2, 5), (3, 6), (1, 5)]
            add_layer_edge!(ms, :a, i, j)
        end
        for (i, j) in [(1, 2), (4, 5), (1, 4)]
            add_layer_edge!(ms, :b, i, j)
        end
        @test compute(WithinLayer(NodeMatch(:grp), 1), ms) == 0.0
        fs = @test_logs (:warn, r"L1.nodematch.grp are at their smallest attainable values") match_mode=:any ergm_multi(
            ms, [LayerEdges(), WithinLayer(NodeMatch(:grp), 1)])
        @test coef(fs)[2] == -Inf && stderror(fs)[2] == 0.0
        # 30 within-layer dyads, 6 within-group ones in layer 1 dropped; 7 ties
        # among the remaining 24
        @test coef(fs)[1] ≈ log(7 / 17) atol = 1e-8
        @test !is_exact(fs)

        # Separation by a COMBINATION of columns (ERGM.jl's `sep2` design lifted
        # into layer 1, with an unrelated layer 2): no single column is at its
        # boundary, yet nodecov.x + nodecov.z perfectly predicts layer 1's
        # ties. R: "The MPLE does not exist!"; here ERGM.jl's asymptote test
        # flags it, the fit is returned unconverged and every honesty channel
        # says so.
        sep = MultilayerNetwork(6; directed=false)
        add_layer!(sep, :a); add_layer!(sep, :b)
        sx, sz = [1, 2, 1, 2, 3, -2], [-1, -2, 3, 0, -3, -1]
        for l in 1:2, v in 1:6
            set_vertex_attribute!(sep.layers[l], :x, v, sx[v])
            set_vertex_attribute!(sep.layers[l], :z, v, sz[v])
        end
        for i in 1:5, j in (i + 1):6
            (sx[i] + sz[i]) + (sx[j] + sz[j]) > 0 && add_layer_edge!(sep, :a, i, j)
        end
        for (i, j) in [(1, 2), (3, 4), (5, 6), (2, 5)]
            add_layer_edge!(sep, :b, i, j)
        end
        sterms = [LayerEdges(1), LayerEdges(2), WithinLayer(NodeCov(:x), 1),
                  WithinLayer(NodeCov(:z), 1)]
        smodel = MultiERGMModel(sterms, sep)
        Xf, y, η0 = ERGMMulti._multi_mple_design(sep, smodel.terms, smodel.offsets, [1, 2, 3, 4])
        @test isempty(ERGM._boundary_columns_iterated(Xf, ones(length(y)), Float64.(y)))
        fsep = @test_logs (:warn, r"the MPLE does not exist \(perfect separation\)") match_mode=:any ergm_multi(sep, sterms)
        @test !fsep.converged && fsep.separated
        @test !is_exact(fsep)
        @test all(isfinite, coef(fsep))         # the last iterate, returned but flagged
        @test any(occursin("does not exist", a) for a in approximations(fsep))
        outs = sprint(show, fsep)
        @test occursin("converged: false", outs) && occursin("MPLE does not exist", outs)
        # The separated case prints R's sentence, not the generic one as well
        logs, _ = Test.collect_test_logs() do
            ergm_multi(sep, sterms)
        end
        @test count(l -> occursin("does not exist", string(l.message)), logs) == 1
        @test !any(l -> occursin("did not converge within", string(l.message)), logs)
        # The unrelated layer's edges coefficient is unaffected in kind: layer 2
        # has 4 ties among 15 dyads and its column is not part of the
        # separating combination
        @test coef(fsep)[2] ≈ log(4 / 11) atol = 1e-6
        # The bootstrap's refits treat a separated replicate as a NaN row
        @test ERGMMulti._multi_mple_fit(sep, smodel.terms, smodel.offsets,
                                        [1, 2, 3, 4]; warn=false).separated

        # The SAME separated design with a non-zero OFFSET is flagged too: the
        # offset contribution is handed to ERGM.jl's asymptote test as a column
        # with a fixed coefficient of 1 (SE 0). It used to be skipped — the fit
        # came back `converged = true, separated = false` with coefficients
        # 20.7 / 25.1 and standard errors in the tens of thousands, and
        # `approximations` listed nothing but the offset note.
        for offs in (Dict(2 => log(4 / 11)), Dict(1 => -1.0), Dict(2 => 0.7, 1 => -3.0))
            fso = @test_logs (:warn, r"the MPLE does not exist \(perfect separation\)") match_mode=:any ergm_multi(
                sep, sterms; offsets=offs)
            @test !fso.converged && fso.separated && !is_exact(fso)
            @test any(occursin("does not exist", a) for a in approximations(fso))
            @test occursin("MPLE does not exist", sprint(show, fso))
            for (k, c) in offs
                @test coef(fso)[k] == c
            end
        end
        # ... while a well-posed offset fit is not: the fixture's per-layer
        # edges with layer 2 pinned
        fok = ergm_multi(fixture(), [LayerEdges(1), LayerEdges(2)]; offsets=Dict(2 => -2.0))
        @test fok.converged && !fok.separated && is_exact(fok)
        # and the golden offset fit (twolayer_ergm_multi.toml) stays unflagged
        # (asserted converged in its own testset)
        # `_multi_separated` itself: the augmented call sees the offset in the
        # linear predictor
        Xo, yo, ηo = ERGMMulti._multi_mple_design(sep, smodel.terms, Dict(2 => log(4 / 11)), [1, 3, 4])
        @test !all(iszero, ηo)
        fo = Networks.logistic_derivatives(Xo, yo; offset=ηo)
        nf = Networks.newton_fit(fo, zeros(3); maxiter=100, tol=1e-8)
        @test ERGMMulti._multi_separated(fo, Xo, yo, ηo, nf.θ, nf.se)
        Xk, yk, ηk = ERGMMulti._multi_mple_design(fixture(), MultiERGMModel([LayerEdges(1), LayerEdges(2)], fixture()).terms,
                                                  Dict(2 => -2.0), [1])
        fk = Networks.logistic_derivatives(Xk, yk; offset=ηk)
        nk = Networks.newton_fit(fk, zeros(1); maxiter=100, tol=1e-8)
        @test !ERGMMulti._multi_separated(fk, Xk, yk, ηk, nk.θ, nk.se)

        # No "compatibility" constructors that default `se_type`,
        # `boot_replicates` or `separated`: the only way to build a result is
        # to pass all eleven fields, so its flags cannot silently disagree
        # with the numbers it carries
        @test fieldcount(MultiERGMResult) == 11
        @test all(mt -> mt.nargs - 1 == fieldcount(MultiERGMResult), methods(MultiERGMResult))
        @test_throws MethodError MultiERGMResult(fsep.model, coef(fsep), stderror(fsep), vcov(fsep),
                                                 loglikelihood(fsep), aic(fsep), bic(fsep), false, :hessian)
        @test_throws MethodError MultiERGMResult(fsep.model, coef(fsep), stderror(fsep), vcov(fsep),
                                                 loglikelihood(fsep), aic(fsep), bic(fsep), false, :hessian, nothing)
    end

    @testset "Loud non-convergence: a rank-deficient design" begin
        # Two copies of the same statistic: the pseudo-Hessian is singular from
        # the first iteration, so Newton has no direction. `Networks.newton_fit`
        # warns that the Hessian is not negative definite and returns NaN
        # standard errors with `converged == false`; `ergm_multi` adds its own
        # non-convergence warning and the result says so everywhere.
        m = MultilayerNetwork(30; directed=true)
        add_layer!(m, :friendship); add_layer!(m, :advice)
        add_layer_edge!(m, :friendship, 1, 2)
        add_layer_edge!(m, :advice, 2, 3)
        logs_d, fd = Test.collect_test_logs() do
            ergm_multi(m, [LayerEdges(1), LayerEdges(1)])
        end
        @test any(l -> occursin("not negative definite", string(l.message)), logs_d)
        @test any(l -> occursin("did not converge within maxiter", string(l.message)), logs_d)
        @test !fd.converged && !fd.separated
        @test all(isnan, stderror(fd))
        @test all(isnan, vcov(fd))
        @test !is_exact(fd)
        @test any(occursin("did not", a) for a in approximations(fd))
        @test occursin("converged: false", sprint(show, fd))
        @test all(isnan, confint(fd))
        # `maxiter=1` on the golden design's terms: the generic sentence, once
        logs, f1 = Test.collect_test_logs() do
            ergm_multi(m, [LayerEdges(1), LayerEdges(2)]; maxiter=1)
        end
        @test !f1.converged && !is_exact(f1)
        @test count(l -> occursin("did not converge within maxiter = 1", string(l.message)), logs) == 1
        @test occursin("did not converge", join(approximations(f1)))
    end

    @testset "Bootstrap replicates without a finite MPLE are excluded (round-3 pattern)" begin
        # A 6-actor two-layer network with 5 triangles: at the MPLE many
        # simulated draws have NO triangle, so `LayerTriangle` sits at its
        # smallest attainable value on those replicates and their refit has no
        # finite MPLE. They must be NaN rows — excluded, warned about once,
        # kept in `boot_replicates` — not "converged" points on the asymptote
        # that inflate every standard error.
        mt = MultilayerNetwork(6; directed=true)
        add_layer!(mt, :a); add_layer!(mt, :b)
        rng0 = Xoshiro(4)
        for l in 1:2, i in 1:6, j in 1:6
            i == j && continue
            rand(rng0) < 0.3 && add_layer_edge!(mt, l, i, j)
        end
        @test compute(LayerTriangle(), mt) == 5.0
        terms = [LayerEdges(), LayerTriangle()]
        kw = (se=:bootstrap, n_boot=40, boot_burnin=200, boot_interval=20)
        logs, boot = Test.collect_test_logs() do
            ergm_multi(mt, terms; kw..., rng=Xoshiro(1))
        end
        excl = [l for l in logs if occursin("bootstrap refits did not converge", string(l.message))]
        @test length(excl) == 1                       # warned ONCE, not per replicate
        @test !any(l -> occursin("attainable values", string(l.message)), logs)   # refits are silent
        @test boot.boot_replicates isa Matrix{Float64}
        @test size(boot.boot_replicates) == (40, 2)
        n_bad = count(isnan, boot.boot_replicates[:, 1])
        @test n_bad > 0
        @test occursin("$n_bad of the 40", string(excl[1].message))
        @test all(isfinite, stderror(boot))
        @test all(isfinite, vcov(boot))
        # The reported covariance IS the empirical covariance of the finite rows
        ok = [all(isfinite, view(boot.boot_replicates, b, :)) for b in 1:40]
        @test vcov(boot) ≈ cov(boot.boot_replicates[ok, :])
        @test any(occursin("$n_bad of the 40 bootstrap refits", a) for a in approximations(boot))
        @test occursin("excluded", sprint(show, boot))
        @test se_method(boot) === :bootstrap
        # Thread-count independence: the replicates are simulated from `rng`
        # before any refit and each refit is deterministic, so `threaded`
        # changes nothing but the wall time
        b_thr = ergm_multi(mt, terms; kw..., rng=Xoshiro(3), threaded=true)
        b_ser = ergm_multi(mt, terms; kw..., rng=Xoshiro(3), threaded=false)
        @test stderror(b_thr) == stderror(b_ser)
        @test vcov(b_thr) == vcov(b_ser)
        @test isequal(b_thr.boot_replicates, b_ser.boot_replicates)
    end

    @testset "The raw sampler refuses masked dyads" begin
        # `simulate_multi_ergm` toggles every within-layer dyad of every layer:
        # a masked dyad would be conditioned on at its face value. The guard is
        # `require_observed` on each layer, naming it; `gof` is covered
        # transitively (a `MultiERGMResult` never holds a masked network).
        m = MultilayerNetwork(5; directed=true)
        add_layer!(m, :a); add_layer!(m, :b)
        add_layer_edge!(m, :a, 1, 2)
        terms = [LayerEdges(), InterlayerDependence(1, 2)]
        θ = [-1.0, 1.0]
        @test length(simulate_multi_ergm(m, terms, θ; n_sim=2, burnin=10, interval=5)) == 2
        set_missing_dyad!(m.layers[1], 3, 4)
        @test_throws ArgumentError simulate_multi_ergm(m, terms, θ; n_sim=1, burnin=10)
        msg = errmsg(() -> simulate_multi_ergm(m, terms, θ; n_sim=1, burnin=10))
        @test occursin("simulate_multi_ergm (layer 1)", msg) && occursin("masked", msg)
        # ... through the prebuilt-model form as well, and for a masked dyad
        # whose face value is a present tie
        model = MultiERGMModel(terms, m)
        @test_throws ArgumentError simulate_multi_ergm(model, θ; n_sim=1, burnin=10)
        clear_missing_dyads!(m.layers[1])
        set_missing_dyad!(m.layers[2], 1, 2)
        @test occursin("layer 2", errmsg(() -> simulate_multi_ergm(m, terms, θ; n_sim=1, burnin=10)))
        clear_missing_dyads!(m.layers[2])
        @test length(simulate_multi_ergm(m, terms, θ; n_sim=1, burnin=10)) == 1
        # The capability the generator prints: no `missing=` keyword, refuse
        @test Networks.missing_policies(simulate_multi_ergm) == (:error,)
        @test Networks.missing_policies(ergm_multi) == (:error,)
        @test !Networks.supports_missing(simulate_multi_ergm)
    end

    # ------------------------------------------------------------------
    # Golden fixture: a REAL statnet `ergm.multi` fit, with provenance (issue #8).
    # test/fixtures/r/twolayer_ergm_multi.R regenerates it.
    #
    # R model:  Networks(list(net1, net2)) ~
    #             N(~edges + nodematch("grp"), ~factor(.NetworkID) - 1)
    # which is per-layer edges and per-layer nodematch — exactly the terms below.
    # Every term is DYAD-INDEPENDENT, so the likelihood factorizes over the
    # (layer, i, j) dyads and both packages compute the same EXACT MLE. That is
    # what licenses the 1e-6 assertions: there is no Monte Carlo to hide in.
    # ------------------------------------------------------------------
    @testset "Golden fixture: statnet ergm.multi on a two-layer network" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "twolayer_ergm_multi.toml"))
        @test g.provenance["ergm_multi_version"] == "0.3.0"

        # Rebuild R's two layers exactly from the frozen edge lists.
        n = Int(g.values["n_actors"])
        grp = String.(g.values["grp"])
        m = MultilayerNetwork(n; directed=true)
        for l in 1:Int(g.values["n_layers"])
            net = network(n; directed=true)
            for v in 1:n
                set_vertex_attribute!(net, :grp, v, grp[v])
            end
            s = Int.(g.values["layer_src"][l])
            d = Int.(g.values["layer_dst"][l])
            for k in eachindex(s)
                add_edge!(net, s[k], d[k])
            end
            add_layer!(m, Symbol("L", l); net=net)
        end
        @test [ne(x) for x in m.layers] == Int.(g.values["edge_counts"])

        terms = AbstractERGMTerm[LayerEdges(1), LayerEdges(2),
                                 WithinLayer(NodeMatch(:grp), 1),
                                 WithinLayer(NodeMatch(:grp), 2)]

        # --- (a) the free fit ------------------------------------------------
        fit = ergm_multi(m, terms)
        @test fit.converged

        # Against ergm.multi as shipped. ergm stops its MPLE glm at the default
        # epsilon = 1e-8; the fixture measures how far that leaves it from the
        # exact optimum (`mple_vs_exact_*`) and sets these tolerances from the
        # measurement rather than from taste.
        @test check_golden(g, "coefficients", fit.coefficients) ||
              error(golden_report(g, "coefficients", fit.coefficients))
        @test check_golden(g, "std_errors", fit.std_errors) ||
              error(golden_report(g, "std_errors", fit.std_errors))

        # Against the SAME estimator taken to convergence (glm at 1e-14). This is
        # the assertion with teeth. Observed: ERGMMulti.jl reproduces it to ~1e-13
        # in the coefficients and ~1e-11 in the standard errors.
        @test check_golden(g, "exact_coefficients", fit.coefficients) ||
              error(golden_report(g, "exact_coefficients", fit.coefficients))
        @test check_golden(g, "exact_std_errors", fit.std_errors) ||
              error(golden_report(g, "exact_std_errors", fit.std_errors))

        # --- (b) OFFSETS -----------------------------------------------------
        # The nodematch coefficients are PINNED at R's offset.coef; only the two
        # edges coefficients are free. This is the half of ergm.multi worth
        # validating: an offset that is dropped from the LINEAR PREDICTOR (rather
        # than merely from the parameter vector) yields a fit that looks fine and
        # is wrong.
        offs = Float64.(g.values["offset_values"])
        fit_off = ergm_multi(m, terms; offsets=Dict(3 => offs[1], 4 => offs[2]))
        @test fit_off.converged
        # The pinned coefficients come back exactly as given, and carry no
        # uncertainty (R reports 0.0 for an offset SE; ERGMMulti.jl reports NaN —
        # both say "this parameter was not estimated").
        @test fit_off.coefficients[3] == offs[1]
        @test fit_off.coefficients[4] == offs[2]
        @test isnan(fit_off.std_errors[3]) && isnan(fit_off.std_errors[4])

        free_off = fit_off.coefficients[1:2]
        @test check_golden(g, "offset_coefficients", free_off) ||
              error(golden_report(g, "offset_coefficients", free_off))
        @test check_golden(g, "exact_offset_coefficients", free_off) ||
              error(golden_report(g, "exact_offset_coefficients", free_off))

        # ...and the offsets are actually IN the linear predictor. If they were
        # silently ignored, the free coefficients would land on the fit of a model
        # with no nodematch term at all — which R also froze. They are nowhere
        # near it (−1.418 vs −1.030), so this testset cannot pass by both sides
        # doing nothing.
        null_coef = Float64.(g.values["no_nodematch_coefficients"])
        @test maximum(abs.(free_off .- null_coef)) > 0.1
    end
    # ------------------------------------------------------------------
    # Second golden fixture: ergm.multi's LAYER terms on the SAME frozen layers
    # (test/fixtures/r/twolayer_layer_terms.R regenerates it): (i) summary()
    # of every term with an ergm.multi counterpart, exact; (ii) a DYAD-
    # DEPENDENT MPLE (L(~edges,~A&B) reads the other layer, L(~mutual,~A) the
    # reverse dyad), which is what licenses the cross-layer and mutual CHANGE
    # STATISTICS against R's — the MPLE is a deterministic function of the
    # change-statistic design, so the two packages must agree to optimizer
    # precision.
    # ------------------------------------------------------------------
    @testset "Golden fixture: ergm.multi layer terms and a dyad-dependent MPLE" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "twolayer_layer_terms.toml"))
        g0 = load_golden(joinpath(@__DIR__, "fixtures", "twolayer_ergm_multi.toml"))
        @test g.provenance["ergm_multi_version"] == "0.3.0"
        @test g.provenance["seed"] == g0.provenance["seed"]
        # One network, two fixtures: the frozen layers are identical
        @test g.values["layer_src"] == g0.values["layer_src"]
        @test g.values["layer_dst"] == g0.values["layer_dst"]
        @test g.values["grp"] == g0.values["grp"]

        n = Int(g.values["n_actors"])
        grp = String.(g.values["grp"])
        m = MultilayerNetwork(n; directed=true)
        for l in 1:Int(g.values["n_layers"])
            net = network(n; directed=true)
            for v in 1:n
                set_vertex_attribute!(net, :grp, v, grp[v])
            end
            s = Int.(g.values["layer_src"][l])
            d = Int.(g.values["layer_dst"][l])
            for k in eachindex(s)
                add_edge!(net, s[k], d[k])
            end
            add_layer!(m, l == 1 ? :A : :B; net=net)
        end
        @test [ne(x) for x in m.layers] == Int.(g.values["edge_counts"])

        # --- (i) the statistics: every term with an ergm.multi counterpart ----
        stat_terms = AbstractERGMTerm[LayerEdges(1), LayerEdges(2), LayerMutual(1),
                                      LayerTriangle(1), InterlayerDependence(1, 2),
                                      MultiplexMutual(1, 2), WithinLayer(GWESP(0.5), 1),
                                      WithinLayer(OStar(2), 2),
                                      WithinLayer(NodeMatch(:grp), 2),
                                      WithinLayer(TwoPath(), 1)]
        @test length(stat_terms) == length(g.values["stat_names"]) == 10
        @test g.values["stat_names"] == ["L(A)~edges", "L(B)~edges", "L(A)~mutual",
                                         "L(A)~triangle", "L(A&B)~edges", "L(A,B)~mutual",
                                         "L(A)~gwesp.OTP.fixed.0.5", "L(B)~ostar2",
                                         "L(B)~nodematch.grp", "L(A)~twopath"]
        vals = [compute(t, m) for t in stat_terms]
        @test check_golden(g, "statistics", vals) ||
              error(golden_report(g, "statistics", vals))
        # ... and through the model's TermSet (materialized terms, same numbers)
        smodel = MultiERGMModel(stat_terms, m)
        @test check_golden(g, "statistics", collect(compute_all(smodel.terms, m)))
        # The cross-layer terms count ORDERED dyads, as R's L(~edges, ~A&B) and
        # mutualL(Ls = list(~A, ~B)) do (checked in the R script on a 4-actor
        # example where ordered and unordered counting differ)
        @test vals[5] == 13.0 && vals[6] == 12.0
        # The gwesp label agrees with R's on a directed layer
        @test name(WithinLayer(GWESP(0.5), 1), m) == "L1.gwesp.OTP.fixed.0.5"

        # --- (ii) the dyad-dependent MPLE -----------------------------------
        terms = [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2), LayerMutual(1)]
        @test g.values["mple_julia_terms"] == ["LayerEdges(1)", "LayerEdges(2)",
                                               "InterlayerDependence(1, 2)", "LayerMutual(1)"]
        fit = ergm_multi(m, terms)
        @test fit.converged && !fit.separated
        @test has_dyad_dependent(fit.model) && !is_exact(fit)
        @test nobs(fit) == Int(g.values["mple_dyad_universe"]) == 2 * n * (n - 1)
        # Against ergm.multi as shipped (its glm stops at epsilon = 1e-8; the
        # fixture measures the slack and sets these tolerances from it)
        @test check_golden(g, "mple_coefficients", fit.coefficients) ||
              error(golden_report(g, "mple_coefficients", fit.coefficients))
        @test check_golden(g, "mple_std_errors", fit.std_errors) ||
              error(golden_report(g, "mple_std_errors", fit.std_errors))
        # Against R's OWN compressed design (ergmMPLE output="matrix") refit at
        # epsilon = 1e-14: the assertion with teeth. Observed: ~9e-8 in the
        # coefficients and ~7e-9 in the standard errors against 1e-6.
        @test check_golden(g, "exact_mple_coefficients", fit.coefficients) ||
              error(golden_report(g, "exact_mple_coefficients", fit.coefficients))
        @test check_golden(g, "exact_mple_std_errors", fit.std_errors) ||
              error(golden_report(g, "exact_mple_std_errors", fit.std_errors))
        # The pseudo-likelihood at R's exact optimum is not above ours: the two
        # packages maximize the same function
        smodel2 = MultiERGMModel(terms, m)
        Xf, y, η0 = ERGMMulti._multi_mple_design(m, smodel2.terms, smodel2.offsets, [1, 2, 3, 4])
        f = Networks.logistic_derivatives(Xf, y; offset=η0)
        ll_r = f(Float64.(g.values["exact_mple_coefficients"]))[1]
        @test loglikelihood(fit) >= ll_r - 1e-7

        # --- (iv) UNDIRECTED layers: the other dyad universe ------------------
        # Sections (i)-(iii) are directed. The undirected branch — unordered
        # within-layer dyads (i < j, nobs = L·n(n−1)/2), `InterlayerDependence`
        # counting unordered co-occurrences, the undirected triangle / gwesp /
        # kstar / gwdegree forms — is pinned against ergm.multi on two
        # undirected 12-actor layers frozen from the same seed stream.
        nu = Int(g.values["und_n_actors"])
        grpu = String.(g.values["und_grp"])
        @test g.values["und_directed"] == false
        mu = MultilayerNetwork(nu; directed=false)
        for l in 1:Int(g.values["und_n_layers"])
            net = network(nu; directed=false)
            for v in 1:nu
                set_vertex_attribute!(net, :grp, v, grpu[v])
            end
            s = Int.(g.values["und_layer_src"][l])
            d = Int.(g.values["und_layer_dst"][l])
            @test all(s .< d)                     # frozen as unordered pairs
            for k in eachindex(s)
                add_edge!(net, s[k], d[k])
            end
            add_layer!(mu, l == 1 ? :A : :B; net=net)
        end
        @test mu isa MultilayerNetwork{false}
        @test [ne(x) for x in mu.layers] == Int.(g.values["und_edge_counts"])
        und_terms = AbstractERGMTerm[LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2),
                                     LayerTriangle(1), WithinLayer(GWESP(0.5), 1),
                                     WithinLayer(NodeMatch(:grp), 2), WithinLayer(Kstar(2), 2),
                                     WithinLayer(GWDegree(0.3), 2)]
        @test g.values["und_stat_names"] == ["L(A)~edges", "L(B)~edges", "L(A&B)~edges",
                                             "L(A)~triangle", "L(A)~gwesp.fixed.0.5",
                                             "L(B)~nodematch.grp", "L(B)~kstar2",
                                             "L(B)~gwdeg.fixed.0.3"]
        @test length(und_terms) == length(g.values["und_stat_julia_terms"]) == 8
        und_vals = [compute(t, mu) for t in und_terms]
        @test check_golden(g, "und_statistics", und_vals) ||
              error(golden_report(g, "und_statistics", und_vals))
        umodel = MultiERGMModel(und_terms, mu)
        @test check_golden(g, "und_statistics", collect(compute_all(umodel.terms, mu)))
        # The undirected labels agree with R's (no OTP on an undirected layer)
        @test name(WithinLayer(GWESP(0.5), 1), mu) == "L1.gwesp.fixed.0.5"
        @test umodel.terms.names[end] == "L2.gwdeg.fixed.0.3"     # R: L(B)~gwdeg.fixed.0.3
        # The dyad-dependent MPLE over the UNORDERED within-layer dyads:
        # ergmMPLE's weights sum to 2 · 12 · 11 / 2 = 132, and so does nobs
        uterms = [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2), LayerTriangle(1)]
        @test g.values["und_mple_julia_terms"] == ["LayerEdges(1)", "LayerEdges(2)",
                                                   "InterlayerDependence(1, 2)", "LayerTriangle(1)"]
        ufit = ergm_multi(mu, uterms)
        @test ufit.converged && !ufit.separated && has_dyad_dependent(ufit.model)
        @test nobs(ufit) == Int(g.values["und_mple_dyad_universe"]) == 2 * nu * (nu - 1) ÷ 2
        @test check_golden(g, "und_mple_coefficients", ufit.coefficients) ||
              error(golden_report(g, "und_mple_coefficients", ufit.coefficients))
        @test check_golden(g, "und_mple_std_errors", ufit.std_errors) ||
              error(golden_report(g, "und_mple_std_errors", ufit.std_errors))
        # Against R's own design refit at epsilon = 1e-14 (observed: ~2e-11 in
        # the coefficients, ~6e-10 in the standard errors, against 1e-6)
        @test check_golden(g, "und_exact_mple_coefficients", ufit.coefficients) ||
              error(golden_report(g, "und_exact_mple_coefficients", ufit.coefficients))
        @test check_golden(g, "und_exact_mple_std_errors", ufit.std_errors) ||
              error(golden_report(g, "und_exact_mple_std_errors", ufit.std_errors))
        umodel2 = MultiERGMModel(uterms, mu)
        Xu, yu, ηu = ERGMMulti._multi_mple_design(mu, umodel2.terms, umodel2.offsets, [1, 2, 3, 4])
        @test length(yu) == 132
        fu = Networks.logistic_derivatives(Xu, yu; offset=ηu)
        @test loglikelihood(ufit) >= fu(Float64.(g.values["und_exact_mple_coefficients"]))[1] - 1e-7
    end

    # ------------------------------------------------------------------
    # WP3 — sampler on the shared kernel, adapters under the conversion
    # contract, precompile workload, CI gates (panel 2026-09, items 4, 7,
    # 18, 28, 29)
    # ------------------------------------------------------------------
    @testset "Sampler is bit-identical to the pre-kernel loop (literal pins)" begin
        # The edge lists below were produced by a verbatim replica of the
        # committed pre-kernel loop (`simulate_multi_ergm` at the 2026-09-12
        # Fable baseline, commit 0a82f9a "added bib citation": layer, i, j
        # drawn in that order, then `log(rand(rng)) < log_accept`) with the
        # same seed; `ERGM.mh_toggle!` draws from the rng in the same order,
        # so the chains coincide toggle for toggle. A change to the proposal,
        # the acceptance rule or the rng order in either package shows up
        # here as a literal mismatch, not as a subtle drift.
        edgelist(s) = [[(Int(src(e)), Int(dst(e))) for e in edges(s.layers[l])] for l in 1:2]

        terms = [LayerEdges(1), LayerEdges(2), LayerMutual(), InterlayerDependence(1, 2)]
        θ = [-0.5, -0.5, 0.8, 1.0]
        draws = simulate_multi_ergm(fixture(), terms, θ; n_sim=3, burnin=500, interval=50,
                                    rng=Xoshiro(7))
        @test edgelist.(draws) == [
            [[(1, 2), (1, 3), (1, 4), (2, 3), (3, 2), (3, 4), (4, 2)],
             [(1, 2), (1, 4), (2, 1), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2)]],
            [[(1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3)],
             [(1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]],
            [[(1, 2), (1, 3), (2, 3), (2, 4), (4, 2)],
             [(1, 2), (1, 3), (2, 1), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2)]]]

        # Undirected: the (j < i) swap in the proposal is part of the pin
        tu = [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2)]
        du = simulate_multi_ergm(fixture_undirected(), tu, [-0.3, -0.3, 1.2]; n_sim=3,
                                 burnin=500, interval=50, rng=Xoshiro(7))
        @test edgelist.(du) == [
            [[(1, 2), (1, 3), (1, 4), (1, 5), (2, 3)],
             [(1, 2), (1, 4), (1, 5), (2, 3), (3, 5)]],
            [[(1, 2), (1, 4), (1, 5), (2, 3), (2, 5), (3, 5)],
             [(1, 2), (1, 4), (1, 5), (2, 3), (2, 5), (4, 5)]],
            [[(1, 2), (1, 3), (1, 5), (2, 5), (3, 5)],
             [(1, 2), (1, 3), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5)]]]
        @test all(!is_directed(s) for s in du)

        # The prebuilt-model form is the same chain
        model = MultiERGMModel(terms, fixture())
        @test edgelist.(simulate_multi_ergm(model, θ; n_sim=3, burnin=500, interval=50,
                                            rng=Xoshiro(7))) == edgelist.(draws)
    end

    @testset "MH step is allocation-free on an 8-term directed model" begin
        # Eight statistics of every kind — per-layer edges, within-layer
        # reciprocity, the two cross-layer terms, a GWESP, an out-star and a
        # two-path lifted into layers — through the generated fill: the
        # change-statistic vector of one move costs exactly 0 bytes, and the
        # sampler adds nothing per step beyond Graphs.jl growing adjacency
        # vectors on accepted additions (bounded as in ERGM.jl's own pin).
        m = fixture()
        terms = [LayerEdges(1), LayerEdges(2), LayerMutual(), InterlayerDependence(1, 2),
                 MultiplexMutual(1, 2), WithinLayer(GWESP(0.5), 1),
                 WithinLayer(OStar(2), 2), WithinLayer(TwoPath(), 1)]
        model = MultiERGMModel(terms, m)
        @test length(model.terms) == 8
        delta = zeros(8)
        # Measured behind the same function barrier the sampler and the
        # design builder use (`model.terms` is an abstractly typed field; a
        # call from an untyped context pays dynamic dispatch, the kernel's
        # closures never do)
        function fill_alloc(dest, ts::TermSet, net, l, i, j)
            ERGMMulti._change_stat_layer_all!(dest, ts, net, l, i, j)
            return @allocated ERGMMulti._change_stat_layer_all!(dest, ts, net, l, i, j)
        end
        @test fill_alloc(delta, model.terms, m, 1, 2, 3) == 0
        @test fill_alloc(delta, model.terms, m, 2, 4, 1) == 0
        # ... and matches the per-term calls
        @test delta == [change_stat_layer(t, m, 2, 4, 1) for t in model.terms.terms]

        θ = [-1.0, -1.0, 0.5, 0.8, 0.3, 0.2, 0.1, -0.1]
        function steps_alloc(steps)
            rng = Xoshiro(1)
            simulate_multi_ergm(model, θ; n_sim=0, burnin=steps, rng=rng)
            return @allocated simulate_multi_ergm(model, θ; n_sim=0, burnin=steps, rng=rng)
        end
        a10 = steps_alloc(10_000)
        a20 = steps_alloc(20_000)
        @test a20 - a10 < 4 * 10_000
        # A model with ≥ 32 statistics would hit Base's Any32 `map` fallback
        # (~30 KB per step) without the generated fill: expand a NodeFactor
        # over an attribute with many levels to get there and pin 0 B.
        big = fixture()
        covs = [Symbol("x", k) for k in 1:13]
        for l in 1:2, (k, a) in enumerate(covs)
            set_vertex_attribute!(big.layers[l], a, Dict(v => Float64(v * k) for v in 1:4))
        end
        wide = vcat(terms, [WithinLayer(NodeCov(a), l) for l in 1:2 for a in covs])
        wmodel = MultiERGMModel(wide, big)
        @test length(wmodel.terms) == 34
        wd = zeros(length(wmodel.terms))
        @test fill_alloc(wd, wmodel.terms, big, 1, 2, 3) == 0
    end

    @testset "Conversion contract: combine_networks, split_by_layer, as_multilayer (item 4)" begin
        # Mask preservation: an unobserved dyad of layer 2 is unobserved at
        # its block position of the combined network, and comes back
        n = 4
        m = fixture()
        set_missing_dyad!(layer_network(m, :advice), 2, 3)
        c = combine_networks(m)
        @test is_missing_dyad(c, n + 2, n + 3)
        @test !is_missing_dyad(c, 2, 3)
        @test n_missing_dyads(c) == 1
        @test ne(c) == 8                      # edges untouched
        @test has_edge(c, n + 2, n + 3)       # the recorded face value (advice 2→3
                                              # is a tie) is carried unchanged; the
                                              # mask says it was never observed
        m2 = split_by_layer(c, n, 2; names=[:friendship, :advice])
        @test is_missing_dyad(layer_network(m2, :advice), 2, 3)
        @test n_missing_dyads(layer_network(m2, :advice)) == 1
        @test n_missing_dyads(layer_network(m2, :friendship)) == 0
        @test all(m2.layers[l].graph == m.layers[l].graph for l in 1:2)
        # ... and the estimator still refuses the masked layer (the contract's
        # guard sits on the fit, the adapter carries the mask to it honestly)
        @test_throws ArgumentError ergm_multi(m2, [LayerEdges()])

        # Several masks across layers, directed and undirected canonical forms
        set_missing_dyad!(layer_network(m, :friendship), 4, 1)
        c2 = combine_networks(m)
        @test n_missing_dyads(c2) == 2 && is_missing_dyad(c2, 4, 1) && !is_missing_dyad(c2, 1, 4)
        mu = fixture_undirected()
        set_missing_dyad!(mu.layers[1], 5, 2)          # stored as (2, 5)
        set_missing_dyad!(mu.layers[2], 3, 1)
        cu = combine_networks(mu)
        @test !is_directed(cu)
        @test is_missing_dyad(cu, 2, 5) && is_missing_dyad(cu, 5, 2)
        @test is_missing_dyad(cu, 5 + 1, 5 + 3)
        @test n_missing_dyads(cu) == 2
        mu2 = split_by_layer(cu, 5, 2)
        @test mu2 isa MultilayerNetwork{false}
        @test collect(missing_dyads(mu2.layers[1])) == [(2, 5)]
        @test collect(missing_dyads(mu2.layers[2])) == [(1, 3)]

        # A masked cross-block dyad is as ill-formed as a cross-block edge
        set_missing_dyad!(c, 1, n + 2)
        msg = errmsg(() -> split_by_layer(c, n, 2))
        @test occursin("masked cross-block dyad (1, 6)", msg)
        @test occursin("structurally empty", msg)
        ce = combine_networks(fixture())
        add_edge!(ce, 1, n + 2)
        @test occursin("cross-block edge (1, 6)", errmsg(() -> split_by_layer(ce, n, 2)))
        @test occursin("one name per layer", errmsg(() -> split_by_layer(ce, n, 2; names=[:a])))

        # report=true: what the block-diagonal network cannot hold is named
        m3 = fixture()
        set_vertex_attribute!(layer_network(m3, :friendship), :grp,
                              Dict(1 => "A", 2 => "A", 3 => "B", 4 => "B"))
        set_edge_attribute!(layer_network(m3, :advice), :w, 1, 2, 2.5)
        set_network_attribute!(layer_network(m3, :advice), :title, "advice wave 1")
        c3, rep = combine_networks(m3; report=true)
        @test c3 isa Network && rep isa ConversionReport
        @test rep.source === :MultilayerNetwork && rep.target === :Network
        @test !is_lossless(rep)
        @test dropped_fields(rep) == [:grp, :w, :title]
        @test occursin("layer 1 (:friendship)", rep.dropped[1].second)
        @test occursin("layer 2 (:advice)", rep.dropped[2].second)
        @test occursin("only :layer and :actor", rep.dropped[1].second)
        @test sort(list_vertex_attributes(c3)) == [:actor, :layer]
        @test isempty(list_edge_attributes(c3)) && isempty(list_network_attributes(c3))
        # `report=false` (the default) returns the network alone, unchanged
        @test combine_networks(m3) isa Network
        @test combine_networks(m3).graph == c3.graph
        # Nothing to drop → lossless
        _, rep0 = combine_networks(fixture(); report=true)
        @test is_lossless(rep0) && isempty(dropped_fields(rep0))
        # The inverse direction reports the combined network's own extras
        set_vertex_attribute!(c3, :zeta, Dict(v => v for v in 1:8))
        set_network_attribute!(c3, :note, "x")
        m4, rep4 = split_by_layer(c3, n, 2; report=true)
        @test m4 isa MultilayerNetwork{true}
        @test rep4.source === :Network && rep4.target === :MultilayerNetwork
        @test dropped_fields(rep4) == [:zeta, :note]     # :layer/:actor are structure, not data
        _, rep5 = split_by_layer(combine_networks(fixture()), n, 2; report=true)
        @test is_lossless(rep5)

        # as_multilayer stores the layers as given: lossless, masks and
        # attributes intact
        a = network(4; directed=true); add_edge!(a, 1, 2); set_missing_dyad!(a, 3, 4)
        set_vertex_attribute!(a, :grp, Dict(1 => "A", 2 => "A", 3 => "B", 4 => "B"))
        b = network(4; directed=true); add_edge!(b, 2, 3)
        ml, repl = as_multilayer([a, b], [:x, :y]; report=true)
        @test is_lossless(repl)
        @test repl.source === :Network && repl.target === :MultilayerNetwork
        @test layer_network(ml, :x) === a                 # not copied
        @test n_missing_dyads(layer_network(ml, :x)) == 1
        @test get_vertex_attribute(layer_network(ml, :x), :grp, 3) == "B"
        @test as_multilayer([a, b], [:x, :y]) isa MultilayerNetwork{true}

        # The trait: none of the three reads a face value
        @test supports_missing(combine_networks)
        @test supports_missing(split_by_layer)
        @test supports_missing(as_multilayer)
        @test missing_policies(combine_networks) == (:error,)   # no `missing=` keyword: nothing to opt into

        # The loops flag travels both ways
        ml_loops = MultilayerNetwork(3; directed=true)
        add_layer!(ml_loops, :a; net=network(3; directed=true, loops=true))
        add_edge!(ml_loops.layers[1], 2, 2)
        cl = combine_networks(ml_loops)
        @test cl.loops && has_edge(cl, 2, 2)
        @test split_by_layer(cl, 3, 1).layers[1].loops
        @test has_edge(split_by_layer(cl, 3, 1).layers[1], 2, 2)
    end

    @testset "CI clone lists are derived from [sources] (items 7, 29)" begin
        # The workflows rebuild the sibling layout by cloning exactly the
        # packages this package sources by path — no more (a stale clone
        # hides a missing dependency), no fewer (the build cannot resolve).
        root = joinpath(@__DIR__, "..")
        project = read(joinpath(root, "Project.toml"), String)
        block = match(r"\[sources\]\n((?:[^\[]*\n)*)", project)
        @test block !== nothing
        sources = [String(mt.captures[1]) for mt in
                   eachmatch(r"^(\w+)\s*=\s*\{\s*path\s*="m, block.captures[1])]
        @test sort(sources) == ["ERGM", "Networks"]
        for wf in ("CI.yml", "Documentation.yml")
            yml = read(joinpath(root, ".github", "workflows", wf), String)
            clone = match(r"for pkg in ([A-Za-z ]+); do", yml)
            @test clone !== nothing
            pkgs = split(strip(clone.captures[1]))
            @test sort(pkgs) == sort(sources)
            # Bottom-up: the foundation is cloned first
            @test pkgs[1] == "Networks"
            @test occursin("git clone --depth 1 --quiet \"\$org/\$pkg.jl\"", yml)
        end
        # One matrix cell runs the tests on four threads so the bootstrap's
        # thread-count-independence pin exercises the threaded path
        ci = read(joinpath(root, ".github", "workflows", "CI.yml"), String)
        @test occursin("JULIA_NUM_THREADS: \${{ (matrix.version == '1' && matrix.os == 'ubuntu-latest') && '4' || '1' }}", ci)
        # The docs environment sources the same siblings by path
        docs = read(joinpath(root, "docs", "Project.toml"), String)
        for pkg in sources
            @test occursin("$pkg = {path = \"../../$pkg.jl\"}", docs)
        end
        @test occursin("ERGMMulti = {path = \"..\"}", docs)
    end

    @testset "Every export has a docstring with a runnable example (criterion 5)" begin
        # Grade-A criterion 5: every export has a docstring with a runnable
        # example. The Documenter build (checkdocs=:exports) checks presence,
        # not content, so walk the docsystem: every ERGMMulti-owned docstring
        # of an exported binding — including the ones ERGMMulti attaches to
        # the shared Networks/StatsAPI generics (`gof`, `coef`, `confint`, ...)
        # — must contain a fenced ```julia block, and every such block must
        # RUN in a fresh module that has done nothing but `using ERGMMulti`
        # (so an example that needs `ERGM`, `Networks`, `Random` or
        # `LinearAlgebra` says so itself). Warnings the examples deliberately
        # provoke (the deprecated `fit_multi_ergm`) go to a null logger.
        # Mirrors ERGM.jl's and Networks.jl's testsets of the same name.
        meta_multi = Base.Docs.meta(ERGMMulti)
        documented_elsewhere(b) = any(haskey(Base.Docs.meta(m), b)
                                      for m in (Networks, ERGM, StatsAPI))
        undocumented = String[]
        missing_example = String[]
        blocks = Tuple{String,String}[]
        for nm in names(ERGMMulti)
            nm === :ERGMMulti && continue
            b = Base.Docs.Binding(ERGMMulti, nm)
            if !haskey(meta_multi, b)
                documented_elsewhere(b) || push!(undocumented, string(nm))
                continue
            end
            has_example = false
            for (_, ds) in meta_multi[b].docs
                txt = ds.text isa AbstractString ? ds.text : join(string.(ds.text), "\n")
                for m in eachmatch(r"```julia\n(.*?)```"s, txt)
                    has_example = true
                    push!(blocks, (string(nm), String(m.captures[1])))
                end
                occursin("```jldoctest", txt) && (has_example = true)
            end
            has_example || push!(missing_example, string(nm))
        end
        @test isempty(undocumented)
        @test isempty(missing_example)
        # ERGMMulti-owned docstrings (with examples) sit on every foreign
        # generic it extends for MultiERGMResult — the StatsAPI surface and
        # `gof` — not only on its own names
        for nm in (:coef, :stderror, :vcov, :confint, :coeftable, :loglikelihood,
                   :nobs, :dof, :aic, :bic, :gof)
            @test haskey(meta_multi, Base.Docs.Binding(ERGMMulti, nm))
        end
        # The undocumented-until-now `layer_names` is documented with an example
        @test any(nm == "layer_names" for (nm, _) in blocks)
        @test length(blocks) >= 39
        for (nm, code) in blocks
            m = Module(Symbol("DocExample_", nm))
            ok = try
                Core.eval(m, :(using ERGMMulti))
                Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                    Core.eval(m, Meta.parseall(code; filename="docstring:$nm"))
                end
                true
            catch err
                println(stderr, "docstring example of $nm failed: ", sprint(showerror, err))
                false
            end
            @test ok
        end
    end

    @testset "Precompile workload is declared (item 18)" begin
        root = joinpath(@__DIR__, "..")
        project = read(joinpath(root, "Project.toml"), String)
        @test occursin("PrecompileTools = \"aea7be01-6a6a-4083-8856-8a6e6704d82a\"", project)
        @test occursin(r"\[compat\][\s\S]*PrecompileTools = \"1\"", project)
        src = read(joinpath(root, "src", "ERGMMulti.jl"), String)
        @test occursin("@setup_workload begin", src) && occursin("@compile_workload begin", src)
        # The workload exercises both directednesses and every documented
        # first-session call
        for needle in ("for _pc_directed in (true, false)", "fit_ergm_multi(_pc_m, _pc_terms)",
                       "simulate_multi_ergm(_pc_m", "gof(_pc_fit;", "se=:bootstrap")
            @test occursin(needle, src)
        end
        # ... and the model it compiles is the documented first model
        @test occursin("WithinLayer(NodeMatch(:grp), 1)", src)
    end
end
