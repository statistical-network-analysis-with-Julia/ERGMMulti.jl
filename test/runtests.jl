using ERGMMulti
using ERGM
using Networks
using Graphs: src, dst
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
                 WithinLayer(TwoPath(), 2), WithinLayer(Kstar(2), 1),
                 WithinLayer(GWESP(0.5), 2), WithinLayer(GWDegree(0.3), 1)]

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
            free = collect(1:length(terms))
            Xf, y, η0 = ERGMMulti._multi_mple_design(mm, terms,
                                                     Dict{Int,Float64}(), free)
            d = ERGM.logistic_derivatives(Xf, y; offset=η0)
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

    @testset "Aliases" begin
        @test fit_multi_ergm === ergm_multi
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

        # The dependent model's robust SEs EXCEED the Hessian ones, on both
        # coefficients — that gap IS the anticonservatism the issue is about
        @test all(stderror(boot) .> stderror(hess))

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
end
