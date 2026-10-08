using ERGMMulti
using ERGM
using NetworkCore
using Graphs: src, dst
using Random
using Statistics
using StatsAPI: StatsAPI
using Test
using Aqua

# Text files read by the tests are compared line by line; a Windows checkout
# (git's core.autocrlf) gives them CRLF endings, so normalise to LF.
_readtext(path) = replace(read(path, String), "\r\n" => "\n")

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

# Run a fresh Julia process and return its exit status, stdout and stderr.
# When the process fails, or its output is not what the caller expects
# (`expect_lines`), its stderr is printed: a subprocess test that fails must
# say why.
function run_fresh(cmd; expect_lines::Union{Nothing, Int}=nothing)
    out, err = IOBuffer(), IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout=out, stderr=err))
    o, e = String(take!(out)), String(take!(err))
    nlines = length(split(strip(o), '\n'))
    if !success(proc) || (expect_lines !== nothing && nlines != expect_lines)
        println(stderr, "fresh Julia process: exit code $(proc.exitcode), ",
                "$nlines line(s) of output", expect_lines === nothing ? "" :
                " ($expect_lines expected)", "; its stdout:\n", o, "\nits stderr:\n", e)
    end
    return (ok=success(proc), out=o, err=e)
end

# The sibling checkouts of the ecosystem layout (the directories its
# `[sources]` name) that sit beside `pkgdir`. None present means a lone
# checkout or a registry install, where `[sources]` is not used and the
# workflows' layout step cannot be run; some present means the layout, where
# a missing one is an error.
sibling_checkouts(pkgdir, siblings) =
    filter(s -> isfile(joinpath(dirname(pkgdir), s, "Project.toml")), collect(siblings))

# The error message of a call that must throw, for `occursin` assertions
function errmsg(f)
    try
        f()
        return ""
    catch e
        return sprint(showerror, e)
    end
end

# The fixtures list each R term's Julia counterpart as a string beside the R
# formula. A string is evaluated only after its expression is checked to be a
# term constructor applied to literals (numbers, `:symbols`, ranges, vectors,
# `:`), so no other code can run from a data file.
const _FIXTURE_CONSTRUCTORS = Set([
    :LayerEdges, :LayerMutual, :LayerTriangle, :WithinLayer, :InterlayerDependence,
    :MultiplexMutual, :Edges, :Mutual, :Triangle, :GWESP, :GWDSP, :GWNSP, :OStar,
    :IStar, :Kstar, :TwoPath, :NodeMatch, :NodeFactor, :NodeMix, :NodeCov, :AbsDiff,
    :IDegree, :ODegree, :Degree, :GWIDegree, :GWODegree, :GWDegree, :MeanDeg,
    :Density, :Concurrent, :DegRange])
function _literal_term_expr(ex)
    ex isa Union{Integer, AbstractFloat, QuoteNode} && return true
    ex === :(:) && return true
    ex isa Expr || return false
    ex.head === :vect && return all(_literal_term_expr, ex.args)
    ex.head === :call || return false
    f = ex.args[1]
    return (f in _FIXTURE_CONSTRUCTORS || f === :(:)) &&
           all(_literal_term_expr, ex.args[2:end])
end
function fixture_term(str::AbstractString)
    ex = Meta.parse(str)
    _literal_term_expr(ex) ||
        error("fixture term $(repr(str)) is not a term constructor applied to literals")
    return Core.eval(@__MODULE__, ex)
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
        r = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)]; method=:mple,
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
    # Allocation regression on the MPLE derivative loop
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
            d = NetworkCore.logistic_derivatives(Xf, y; offset=η0)
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

        r = ergm_multi(draws[1], terms; method=:mple)
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
        ro = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)]; method=:mple,
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
        # NetworkCore.jl coefficient printer (R-style columns, signif codes)
        r_ind = ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
        out_ind = sprint(show, r_ind)
        @test !occursin("dyad-dependent", out_ind)
        @test occursin("Pr(>|z|)", out_ind)
        @test occursin("Std.Error", out_ind)
        @test count("Signif. codes:", out_ind) == 1

        # Dyad-dependent formula → pseudo-likelihood warning
        r_dep = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)]; method=:mple)
        out = sprint(show, r_dep)
        @test occursin("dyad-dependent", out)
        @test occursin("pseudolikelihood", out)
        # The default fit withholds the naive Wald inference and says why ...
        @test occursin("not reported (NaN)", out) && occursin("under-cover", out)
        # ... and an explicit se=:hessian prints R's table with the warning
        out_h = sprint(show, ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)]; method=:mple,
                                        se=:hessian))
        @test occursin("anticonservative", out_h)
    end

    @testset "Goodness of fit" begin
        m = fixture()
        r = ergm_multi(m, [LayerEdges(1), LayerEdges(2)])

        g = gof(r; n_sim=30, burnin=300, interval=30, rng=Random.Xoshiro(41))

        # gof extends NetworkCore.jl's shared generic and returns the shared
        # GOFResult container
        @test ERGMMulti.gof === NetworkCore.gof
        @test g isa GOFResult
        @test n_simulations(g) == 30
        @test length(g.statistics) == 9     # + multiplexity, idegree/odegree/esp per layer

        # Panel 1: the fitted terms' statistics
        stats = g.statistics[1]
        @test stats.name == "model statistics"
        @test stats.labels == ["L(friendship)~edges", "L(advice)~edges"]
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

    @testset "Entry points: fit_ergm_multi and ergm_multi; R's default estimator" begin
        # The harmonised `fit_<model>` name is a `const` alias of the R name;
        # the never-released `fit_multi_ergm` alias is gone (see the docs'
        # "Renamed and removed names" table)
        @test fit_ergm_multi === ergm_multi
        @test !isdefined(ERGMMulti, :fit_multi_ergm)
        @test !(:fit_multi_ergm in names(ERGMMulti))
        m = fixture()
        r = ergm_multi(m, [LayerEdges(1)])
        @test coef(fit_ergm_multi(m, [LayerEdges(1)])) == coef(r)

        # method=:auto (the default) follows R: the MPLE, which is the exact
        # MLE, when no term is dyad-dependent; the MCMLE otherwise
        @test r.method === :mple && r.mcmc === nothing && is_exact(r)
        dep = [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2)]
        fd = ergm_multi(m, dep; n_samples=64, mcmle_maxiter=3, bridge_rungs=2,
                        rng=Xoshiro(1))
        @test fd.method === :mcmle && fd.mcmc !== nothing && se_method(fd) === :fisher
        @test ERGM.resolve_method(:auto, MultiERGMModel(dep, m); context="ergm_multi") === :mcmle
        @test ERGM.resolve_method(:auto, MultiERGMModel([LayerEdges(1)], m)) === :mple
        @test occursin("Method: mcmle (Monte-Carlo maximum likelihood)", sprint(show, fd))
        fm = ergm_multi(m, dep; method=:mple)
        @test fm.method === :mple && fm.inference_withheld
        @test occursin("Method: mple (maximum pseudo-likelihood: an approximation under " *
                       "dyadic dependence; the default method=:auto fits the MCMLE here, " *
                       "as R does)", sprint(show, fm))
        @test occursin("Method: mple (maximum pseudo-likelihood, which is the likelihood: " *
                       "the formula is dyad-independent)", sprint(show, r))
        rx = ergm_multi(m, [LayerEdges(1)]; method=:mcmle)
        @test rx.method === :mcmle && rx.mcmc === nothing && coef(rx) == coef(r)
        @test occursin("Method: mcmle (maximum likelihood, exact", sprint(show, rx))
        msg = errmsg(() -> ergm_multi(m, dep; method=:cmle))
        @test occursin("unknown estimation method :cmle", msg) && occursin(":auto", msg)

        # A keyword of the other estimator is refused in words, naming the one
        # that takes it
        msg = errmsg(() -> ergm_multi(m, dep; se=:bootstrap))
        @test occursin("keyword `se` is not accepted by method=:mcmle", msg) &&
              occursin("method=:auto chose :mcmle because the formula is dyad-dependent", msg) &&
              occursin("pass method=:mple explicitly", msg)
        msg = errmsg(() -> ergm_multi(m, [LayerEdges(1)]; n_samples=64))
        @test occursin("keyword `n_samples` is not accepted by method=:mple", msg) &&
              occursin("dyad-independent", msg) && occursin("pass method=:mcmle explicitly", msg)
        msg = errmsg(() -> ergm_multi(m, dep; method=:mcmle, n_boot=5, se=:hessian))
        @test occursin("`n_boot`, `se`", msg) && !occursin("method=:auto chose", msg)
        msg = errmsg(() -> ergm_multi(m, dep; method=:mple, nonsense=1))
        @test occursin("`nonsense`", msg) && occursin("?ergm_multi", msg)
        # Keyword vocabulary: maxiter / rng / n_sim, read from the estimators
        @test hasmethod(ergm_multi, Tuple{MultilayerNetwork, Any})
        kmple = ERGMMulti._multi_keywords(Val(:mple))
        kmcmle = ERGMMulti._multi_keywords(Val(:mcmle))
        @test all(in(kmple), (:maxiter, :tol, :rng, :se, :n_boot, :boot_burnin,
                              :boot_interval, :threaded))
        @test all(in(kmcmle), (:maxiter, :tol, :rng, :n_samples, :burnin, :interval,
                               :mcmle_maxiter, :termination, :init))
        @test !(:warn in kmcmle) && !(:method in kmple)
        @test all(k -> k in kmple && k in kmcmle, ERGMMulti._SHARED_KEYWORDS)
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
        dep = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)]; method=:mple)
        @test objective(dep) == :pseudolikelihood
        @test !is_exact(dep)
        @test any(occursin("anticonservative", a) for a in approximations(dep))

        # `show`'s prose caveat and the protocol are driven by one predicate
        @test occursin("pseudolikelihood", sprint(show, dep))
        @test !occursin("Warning", sprint(show, indep))

        # Offsets are fixed, not estimated — the protocol says so
        off = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)]; method=:mple,
                         offsets=Dict(1 => -1.0))
        @test any(occursin("offset", a) for a in approximations(off))
    end
    @testset "Robust standard errors: se=:bootstrap" begin
        # Issue #9 / ERGMMulti#1: the within-layer MPLE reported inverse-
        # pseudo-Hessian SEs with no robust alternative, and they are
        # anticonservative whenever a term is dyad-dependent over the (layer, i, j)
        # dyad universe. `se=:bootstrap` adds a parametric bootstrap (simulate at
        # θ̂ with `simulate_multi_ergm`, refit, empirical covariance) on the ONE
        # shared `NetworkCore.bootstrap_cov` loop, with the same API as `ERGM.mple`.
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

        hess = ergm_multi(m, terms; method=:mple)
        boot = ergm_multi(m, terms; method=:mple, se=:bootstrap, n_boot=60,
                          rng=MersenneTwister(11))

        # The bootstrap replaces the COVARIANCE, not the point estimate
        @test coef(boot) == coef(hess)
        @test loglikelihood(boot) == loglikelihood(hess)
        @test aic(boot) == aic(hess)
        @test stderror(boot) != stderror(hess)
        @test vcov(boot) != vcov(hess)
        @test all(isfinite, stderror(boot))

        # Reproducible under a fixed rng
        boot2 = ergm_multi(m, terms; method=:mple, se=:bootstrap, n_boot=60,
                           rng=MersenneTwister(11))
        @test stderror(boot2) == stderror(boot)
        @test vcov(boot2) == vcov(boot)
        @test stderror(ergm_multi(m, terms; method=:mple, se=:bootstrap, n_boot=60,
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
        @test occursin("under-cover", out_h)       # the default withholds z/p and says why
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
        off = ergm_multi(m, terms; method=:mple, offsets=Dict(1 => -1.5), se=:bootstrap,
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
        @test_throws ArgumentError ergm_multi(m, terms; method=:mple, se=:sandwich)
        @test_throws ArgumentError ergm_multi(m, terms; method=:mple, se=:bootstrap, n_boot=1)
    end

    @testset "MultilayerNetwork{D}: directedness is a type parameter" begin
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

    @testset "Model construction validates the formula" begin
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
        @test ergm_multi(m, [LayerEdges(), WithinLayer(Mutual(), 1)]; method=:mple) isa MultiERGMResult
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
                                     "(:friendship, :advice); terms: L(friendship)~edges + L(advice)~edges"
        @test sprint(show, MultiERGMModel([LayerEdges(1), LayerEdges(2)], m;
                                          offsets=Dict(2 => -2.0))) ==
              "MultiERGMModel{true}: 4 actors, 2 directed layers (:friendship, :advice); " *
              "terms: L(friendship)~edges + L(advice)~edges; offsets: 2 => -2.0"
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
        @test model.terms.names == ["L((L1,L2))~edges", "L(L1)~nodematch.grp", "L(L2)~nodecov.w"]
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
        @test mx.terms.names == ["L(L1)~nodefactor.cls.b", "L(L1)~nodefactor.cls.c", "L(L2)~edges"]
        @test mx.offsets == Dict(3 => -1.0)
        fx = ergm_multi(mx)
        @test coef(fx)[3] == -1.0 && isnan(stderror(fx)[3])
        @test coeftable(fx).names[3] == "offset(L(L2)~edges)"
        # An offset on the expanding term itself is refused with the fix
        msg = errmsg(() -> MultiERGMModel([WithinLayer(NodeFactor(:cls), 1), LayerEdges(2)], m;
                                          offsets=Dict(1 => 0.5)))
        @test occursin("expands into 2 statistics", msg) && occursin("level=", msg)
        mu = fixture_undirected()
        md = MultiERGMModel([LayerEdges(), WithinLayer(Degree(0:1), 1)], mu)
        @test md.terms.names == ["L((friendship,advice))~edges", "L(friendship)~degree0",
                                 "L(friendship)~degree1"]
        @test length(simulate_multi_ergm(md, [-1.0, 0.0, 0.0]; n_sim=1, burnin=5, interval=1)) == 1
    end

    @testset "Direction-aware coefficient labels: name(term, m)" begin
        m = fixture()
        mu = fixture_undirected()
        @test name(WithinLayer(GWESP(0.5), 1), m) == "L(friendship)~gwesp.OTP.fixed.0.5"
        @test name(WithinLayer(GWESP(0.5), 1), mu) == "L(friendship)~gwesp.fixed.0.5"
        @test name(WithinLayer(GWDSP(0.5), 2), m) == "L(advice)~gwdsp.OTP.fixed.0.5"
        @test name(WithinLayer(Edges(), 2), m) == "L(advice)~edges"
        # Without a network the layers are named by index, as R names an
        # unnamed layer list
        @test name(WithinLayer(GWESP(0.5), 1)) == "L(1)~gwesp.fixed.0.5"
        @test name(LayerEdges(1)) == "L(1)~edges"
        @test name(LayerEdges([1, 2])) == "L((1,2))~edges"
        @test name(LayerEdges()) == "L((:))~edges"
        @test name(InterlayerDependence(2, 1)) == "L(2&1)~edges"
        @test name(MultiplexMutual(1, 2)) == "L(1,2)~mutual"
        @test name(LayerEdges(1), m) == "L(friendship)~edges"
        @test name(LayerEdges(), m) == "L((friendship,advice))~edges"
        @test_throws ArgumentError name(WithinLayer(Edges(), 3), m)
        @test_throws ArgumentError name(LayerEdges(3), m)
        @test_throws ArgumentError name(InterlayerDependence(1, 3), m)
        # ... and the fitted table carries the same labels as ERGM.jl / R would
        # (the 4-actor fixture has no shared partner, so gwesp sits at its
        # boundary and the fit warns as R does — the labels are the point here)
        fit = @test_logs (:warn, r"attainable") match_mode=:any ergm_multi(
            m, [LayerEdges(1), WithinLayer(GWESP(0.5), 1)]; method=:mple)
        @test coeftable(fit).names == ["L(friendship)~edges", "L(friendship)~gwesp.OTP.fixed.0.5"]
        @test fit.model.terms.names == coeftable(fit).names
        fit_u = ergm_multi(mu, [LayerEdges(1), WithinLayer(GWESP(0.5), 1)]; method=:mple)
        @test coeftable(fit_u).names == ["L(friendship)~edges", "L(friendship)~gwesp.fixed.0.5"]
        g = gof(fit_u; n_sim=5, burnin=20, interval=5, rng=Xoshiro(2))
        @test g.statistics[1].labels == ["L(friendship)~edges", "L(friendship)~gwesp.fixed.0.5"]
        ef = @test_logs (:warn, r"attainable") match_mode=:any ergm_multi(
            m, [LayerEdges(1), WithinLayer(GWESP(0.5), 1)]; method=:mple)
        @test sprint(show, ef) == sprint(show, fit)
    end

    @testset "StatsAPI surface is complete" begin
        m = fixture()
        hess = ergm_multi(m, [LayerEdges(1), LayerEdges(2)])
        boot = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)]; method=:mple,
                          se=:bootstrap, n_boot=12, boot_burnin=100, boot_interval=20,
                          rng=Xoshiro(5))
        off = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)]; method=:mple,
                         offsets=Dict(1 => -1.0), se=:hessian)
        for fit in (hess, boot, off)
            @test check_statsapi(fit; strict=true) !== nothing
            @test all(check_statsapi(fit))
            @test all(values(check_statsapi(fit;
                required=(NetworkCore.STATSAPI_VERBS..., :coefnames), strict=true)))
            @test coefnames(fit) == coeftable(fit).names
            @test coefnames(fit) !== coefnames(fit)          # a copy each time
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
        @test coeftable(off).names == ["offset(L((friendship,advice))~edges)",
                                       "L(friendship&advice)~edges"]
        @test all(isnan, confint(off)[1, :])
        @test isnan(coeftable(off)[1].p_value) && isnan(coeftable(off)[1].z_value)
        @test !isnan(coeftable(off)["L(friendship&advice)~edges"].p_value)
        @test dof(off) == 1
        @test occursin("offset(L((friendship,advice))~edges)", sprint(show, off))
        # z → p is the shared NetworkCore helper (floored, NaN-aware)
        @test coeftable(hess).p_values == z_pvalues(coef(hess), stderror(hess)).p
        # `coefnames` is the StatsAPI binding (one name under `using StatsBase`
        # or any other model package), R's labels with the offset row wrapped
        @test ERGMMulti.coefnames === StatsAPI.coefnames === NetworkCore.coefnames
        @test coefnames(off)[1] == "offset(L((friendship,advice))~edges)"
    end

    @testset "No private cross-package reach-ins" begin
        src = _readtext(joinpath(@__DIR__, "..", "src", "ERGMMulti.jl"))
        # No `ERGM._x` / `NetworkCore._x` anywhere in the source: ERGM.jl's
        # building blocks come from its extension API, `ERGM.Extension`
        # (semver-covered), and an underscore says private
        reach = [m.match for mod in (:ERGM, :NetworkCore)
                 for m in eachmatch(Regex("\\b$(mod)\\._\\w+"), src)]
        @test isempty(reach)
        ext = Set(Symbol(m.captures[1]) for m in eachmatch(r"\bERGM\.Extension\.(\w+)", src))
        for nm in (:boundary_columns, :mple_fit_design, :extreme_statistics,
                   :attainable_range, :validate_formula, :materialize, :mcmle_solve,
                   :bridge_integrate, :mcmc_defaults)
            @test nm in ext
        end
        @test all(nm -> Base.isexported(ERGM.Extension, nm), ext)
        # The layer terms' ranges are methods of ERGM's generic; the MPLE's
        # warnings are `mple_fit_design`'s, worded for ergm.multi through its
        # `note` keyword, and its separation verdict is the one the fit
        # returns: no local warning copies, no second verdict
        @test hasmethod(ERGM.Extension.attainable_range, Tuple{LayerEdges, MultilayerNetwork})
        @test hasmethod(ERGM.Extension.attainable_range, Tuple{WithinLayer, MultilayerNetwork})
        for gone in (:_warn_multi_boundary, :_warn_multi_aliased, :_warn_multi_separated,
                     :_multi_attainable_range, :_multi_extreme_statistics, :_multi_extreme_side)
            @test !isdefined(ERGMMulti, gone)
        end
        @test !occursin("logistic_separation(", src)
        @test keys(ERGMMulti._MULTI_MPLE_NOTES) ==
              (:boundary, :not_varying, :linear_dependence, :separation)
        # No private z→p / Newton / dyad-dependence copies: the shared bindings
        @test !isdefined(ERGMMulti, :_has_dyad_dependent)
        @test !isdefined(ERGMMulti, :_z_pvalues)
        @test ERGMMulti.has_dyad_dependent === ERGM.has_dyad_dependent
        @test ERGMMulti.newton_fit === NetworkCore.newton_fit
        @test ERGMMulti.logistic_derivatives === NetworkCore.logistic_derivatives
        @test ERGMMulti.z_pvalues === NetworkCore.z_pvalues
        @test ERGMMulti.check_se === NetworkCore.check_se
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

    @testset "Sampler runs on the shared Metropolis kernel" begin
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

        # Dyad-scaled defaults come from the ONE rule (ERGM.Extension.mcmc_defaults)
        d = ERGM.Extension.mcmc_defaults(ERGMMulti._n_within_dyads(m))
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
        fit = @test_logs (:warn, r"L\(a&b\)~edges are at their smallest attainable values") match_mode=:any ergm_multi(
            m, [LayerEdges(), InterlayerDependence(1, 2)]; method=:mple)
        # The default for this dyad-dependent formula is the MCMLE, which has
        # no finite estimate here either: as R ergm's drop=TRUE it fixes the
        # coefficient at -Inf, holds the statistic at 0 in the sampler and
        # estimates the rest (it used to refuse); drop=false refuses
        dmle = @test_logs (:warn, r"L\(a&b\)~edges are at their smallest attainable values.*the sampler never moves the statistic off its observed bound") match_mode=:any ergm_multi(
            m, [LayerEdges(), InterlayerDependence(1, 2)]; rng=Xoshiro(4))
        @test dmle.method === :mcmle && dmle.converged
        @test coef(dmle)[2] == -Inf && stderror(dmle)[2] == 0.0
        # every draw kept the dropped statistic at its observed 0
        @test all(==(0.0), [compute(InterlayerDependence(1, 2), s)
                            for s in simulate_multi_ergm(dmle.model,
                                                         [coef(dmle)[1], -ERGMMulti._HOLD_AT_BOUND];
                                                         n_sim=20, rng=Xoshiro(1))])
        msg = errmsg(() -> ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)]; drop=false))
        @test occursin("L(a&b)~edges", msg) && occursin("drop=false", msg) &&
              occursin("not implemented", msg)
        # The sentence is R ergm's, but its closing parenthesis tells the
        # ergm.multi migrant the truth: ergm.multi 0.3.0 does NOT drop (it
        # warns "The MPLE does not exist!" and returns a finite value — the
        # fixture assertions below), so the warning must not say "R ergm
        # reports the same" (ERGM.jl's `warn_boundary` wording, which is
        # true of `ergm` and false of `ergm.multi`)
        blogs, _ = Test.collect_test_logs() do
            ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)]; method=:mple)
        end
        bmsg = only([string(l.message) for l in blogs if occursin("attainable", string(l.message))])
        @test startswith(bmsg, "ergm_multi: observed statistic(s) L(a&b)~edges are at their smallest attainable values. Their coefficients will be fixed at -Inf")
        @test occursin("R ergm's drop=TRUE", bmsg) &&
              occursin("ergm.multi 0.3.0 warns \"The MPLE does not exist!\"", bmsg) &&
              occursin("returns a finite value", bmsg) && occursin("estimation guide", bmsg)
        @test !occursin("R ergm reports the same", bmsg)
        # ... one sentence, ERGM's, with the parenthesis passed through
        # `mple_fit_design`'s `note=`
        @test occursin("note=_MULTI_MPLE_NOTES",
                       _readtext(joinpath(@__DIR__, "..", "src", "ERGMMulti.jl")))
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
        # (the default fit of this dyad-dependent model withholds intervals;
        # the explicit naive opt-in reports the degenerate one)
        @test_throws ArgumentError confint(fit)
        fit_h = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            ergm_multi(fit.model; method=:mple, se=:hessian)
        end
        @test isequal(confint(fit_h)[2, :], [-Inf, -Inf])
        # The same design with the dependence term absent agrees on the edges
        # coefficient only if it also drops the touched rows — it does not, so
        # the two differ (the drop is a real refit, not a relabelling)
        plain = ergm_multi(m, [LayerEdges()])
        @test coef(plain)[1] ≈ log(12 / 48) atol = 1e-8
        @test coef(plain)[1] != coef(fit)[1]

        # A coefficient of -Inf cannot be passed to the sampler (θ'Δ = -Inf·0 =
        # NaN would reject every proposal: `simulate_multi_ergm` refuses it,
        # naming the term). `gof` of the dropped fit simulates the fitted
        # model instead: the statistic held at its observed bound by the
        # finite stand-in, every draw has L(a&b)~edges = 0, and the chain
        # moves (it used to report p = 1.0 everywhere on copies of the data)
        gd = gof(fit; n_sim=20, burnin=200, interval=20, rng=Xoshiro(3))
        @test all(==(0.0), gd.statistics[1].simulated[:, 2])
        @test length(unique(gd.statistics[1].simulated[:, 1])) > 1
        @test_throws ArgumentError simulate_multi_ergm(fit.model, coef(fit); n_sim=1)
        @test_throws ArgumentError simulate_multi_ergm(m, [LayerEdges(), InterlayerDependence(1, 2)],
                                                       [-2.1, -Inf]; n_sim=1)
        @test_throws ArgumentError simulate_multi_ergm(m, [LayerEdges(), InterlayerDependence(1, 2)],
                                                       [NaN, -2.5]; n_sim=1)
        msg = errmsg(() -> simulate_multi_ergm(m, [LayerEdges(), InterlayerDependence(1, 2)],
                                               [NaN, Inf]; n_sim=1))
        @test occursin("simulate_multi_ergm: every coefficient must be finite", msg) &&
              occursin("L((a,b))~edges = NaN", msg) && occursin("L(a&b)~edges = Inf", msg)
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
        bfit = @test_logs (:warn, r"L\(a&b\)~edges are at their smallest attainable values") match_mode=:any ergm_multi(
            m, bterms; method=:mple)
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
        boot = ergm_multi(mb, [LayerEdges(), InterlayerDependence(1, 2)]; method=:mple,
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


    @testset "Golden fixture: boundary statistics, R's drop (ergm.multi's offset spelling)" begin
        # Three designs with one statistic at the bottom of its attainable
        # range (test/fixtures/r/boundary_multi.R). ergm.multi 0.3.0 does not
        # drop at its defaults (NA, or a finite value where its estimation
        # stopped — frozen below as the documented divergence); R ergm's drop
        # written out as `offset(L(...))` at -Inf is a model ergm.multi does
        # fit, and it is what ERGMMulti.jl's default fit must reproduce.
        g = load_golden(joinpath(@__DIR__, "fixtures", "boundary_multi.toml"))
        @test g.provenance["ergm_multi_version"] == "0.3.0"
        v(key) = Float64.(g.values[key])
        function build(key)
            m = MultilayerNetwork(Int(g.values["$(key)_n_actors"]);
                                  directed=g.values["$(key)_directed"])
            add_layer!(m, :A); add_layer!(m, :B)
            for (l, L) in ((1, "A"), (2, "B"))
                for (i, j) in zip(g.values["$(key)_layer_$(L)_src"], g.values["$(key)_layer_$(L)_dst"])
                    add_layer_edge!(m, l, Int(i), Int(j))
                end
                if haskey(g.values, "$(key)_g")
                    for (vtx, lev) in enumerate(g.values["$(key)_g"])
                        set_vertex_attribute!(m.layers[l], :g, vtx, lev)
                    end
                end
            end
            return m
        end
        # The R terms' Julia counterparts, written out (the fixture lists the
        # same spellings next to its R formulas; asserted equal below)
        julia_terms = Dict(
            "a" => ["LayerEdges(1)" => LayerEdges(1), "LayerEdges(2)" => LayerEdges(2),
                    "LayerTriangle(1)" => LayerTriangle(1)],
            "b" => ["LayerEdges(1)" => LayerEdges(1), "LayerEdges(2)" => LayerEdges(2),
                    "InterlayerDependence(1, 2)" => InterlayerDependence(1, 2),
                    "LayerMutual(1)" => LayerMutual(1)],
            "c" => ["LayerEdges(1)" => LayerEdges(1), "LayerEdges(2)" => LayerEdges(2),
                    "WithinLayer(NodeMatch(:g), 1)" => WithinLayer(NodeMatch(:g), 1),
                    "LayerMutual(1)" => LayerMutual(1)])
        n_seeds = length(g.values["seeds"])
        band_sd = g.tolerance["band_sd"]
        for key in ("a", "b", "c")
            m = build(key)
            @test g.values["$(key)_julia_terms"] == first.(julia_terms[key])
            terms = last.(julia_terms[key])
            k = Int(g.values["$(key)_boundary_index"])
            fin = [j for j in eachindex(terms) if j != k]
            model = MultiERGMModel(terms, m)
            @test model.terms.names == g.values["$(key)_stat_names"]      # R's labels
            @test check_golden(g, "$(key)_statistics", compute_all(model.terms, m))
            @test ERGM.Extension.extreme_statistics(model.terms, m) == [(k, :min)]

            # What ergm.multi does at its defaults: no drop
            if key == "a"
                @test g.values["a_r_mple_boundary_na"] && g.values["a_r_mple_not_varying_warning"]
            else
                @test g.values["$(key)_r_mple_not_exist_warning"]
                @test g.values["$(key)_r_mple_boundary"] < -10
            end
            @test isfinite(g.values["$(key)_r_mcmle_boundary"])

            # The MPLE: R's drop, the finite coefficients at the exact limit
            # (case (a) used to stop at 0 for every coefficient: the triangle
            # column is all zeros, so only the attainable range shows the bound)
            mp = @test_logs (:warn, r"are at their smallest attainable values") match_mode=:any ergm_multi(
                model; method=:mple)
            @test mp.converged
            @test coef(mp)[k] == -Inf && stderror(mp)[k] == 0.0
            @test check_golden(g, "$(key)_exact_limit", coef(mp)[fin]) ||
                  error(golden_report(g, "$(key)_exact_limit", coef(mp)[fin]))
            @test dof(mp) == length(fin)
            @test bic(mp) ≈ -2 * loglikelihood(mp) + length(fin) * log(g.values["$(key)_exact_limit_dyads"]) atol = 1e-10
            @test occursin("drop=false",
                           errmsg(() -> ergm_multi(model; method=:mple, drop=false)))

            # The default fit is the MCMLE (dyad-dependent formula). It drops as
            # R ergm does, warning so, and is R's offset fit within R's own
            # Monte-Carlo resolution
            band = band_sd .* v("$(key)_coef_sd") .* sqrt(1 + 1 / n_seeds)
            se_band = band_sd .* v("$(key)_se_sd") .* sqrt(1 + 1 / n_seeds)
            for seed in (1, 2)
                f = @test_logs (:warn, r"ergm.multi 0.3.0 does not drop") match_mode=:any ergm_multi(
                    model; rng=Xoshiro(seed))
                @test f.converged && f.method === :mcmle && f.mcmc !== nothing
                @test coef(f)[k] == -Inf && stderror(f)[k] == 0.0
                @test coeftable(f)[k].p_value == 0.0
                @test all(abs.(coef(f)[fin] .- v("$(key)_coef_mean")) .<= band)
                @test all(abs.(stderror(f)[fin] .- v("$(key)_se_mean")) .<= se_band)
                @test maximum(f.mcmc.mc_std_errors ./ stderror(f)[fin]) <= 0.10
                @test dof(f) == length(fin)
                @test f.mcmc.estimated == fin
                @test any(occursin("the sampler held the statistic at its observed bound", a)
                          for a in approximations(f))
                if key == "c"
                    # A dyad-independent statistic held at its bound: the
                    # log-likelihood is computed, and is R's
                    ll_band = band_sd * sqrt(2) * g.values["c_loglik_sd"]
                    @test abs(loglikelihood(f) - g.values["c_loglik_mean"]) <= ll_band
                    @test f.mcmc.loglik_note === nothing
                else
                    @test isnan(loglikelihood(f)) && isnan(aic(f)) && isnan(bic(f))
                    @test any(occursin("log-likelihood, AIC and BIC not computed", a)
                              for a in approximations(f))
                    @test occursin("log-likelihood, AIC and BIC not computed", sprint(show, f))
                end
                @test all(values(NetworkCore.check_statsapi(f;
                    required=(NetworkCore.STATSAPI_VERBS..., :coefnames), strict=true)))
            end
            # Strict mode: drop=false refuses before any start, with no warning
            logs, msg = Test.collect_test_logs() do
                errmsg(() -> ergm_multi(model; drop=false))
            end
            @test isempty(logs)
            @test occursin("drop=false", msg) && occursin(g.values["$(key)_stat_names"][k], msg) &&
                  occursin("smallest attainable", msg)
        end

        # The within-layer spelling of (a)'s triangle is the same model
        m = build("a")
        wt = ergm_multi(m, [LayerEdges(1), LayerEdges(2), WithinLayer(Triangle(), 1)];
                        method=:mple)
        @test coef(wt)[3] == -Inf
        @test check_golden(g, "a_exact_limit", coef(wt)[1:2])
        # ... and a pooled triangle sits at its bound only when every pooled
        # layer does (layer B has triangles)
        pooled = MultiERGMModel([LayerEdges(1), LayerEdges(2), WithinLayer(Triangle(), [1, 2])], m)
        @test compute(pooled.terms.terms[3], m) > 0
        @test isempty(ERGM.Extension.extreme_statistics(pooled.terms, m))
        # The pooled range is the sum of the per-layer ranges ERGM.jl declares
        # for the term on each layer, through this package's methods of the
        # extension API's generic (no range table of its own)
        @test ERGM.Extension.attainable_range(pooled.terms.terms[3], m) == (0.0, Inf)
        @test ERGM.Extension.attainable_range(WithinLayer(Edges(), [1, 2]), m) ==
              (0.0, 2.0 * ERGM.Extension.attainable_range(Edges(), m.layers[1])[2])
        @test ERGM.Extension.attainable_range(LayerTriangle(1), m) == (0.0, Inf)
        @test !isdefined(ERGMMulti, :_multi_attainable_range)
        @test !isdefined(ERGMMulti, :_multi_extreme_statistics)
    end

    @testset "Boundary statistics: README's single-edge design, separation, bootstrap refusal" begin
        # The README's pre-0.2 Quick Start: 30 actors, ONE friendship tie and
        # an empty advice layer, fit with pooled edges + cross-layer
        # dependence. No dyad is tied in both layers, so L(friendship&advice)~edges sits at
        # its smallest attainable value: R's drop, out loud.
        m = MultilayerNetwork(30; directed=true)
        add_layer!(m, :friendship); add_layer!(m, :advice)
        add_layer_edge!(m, :friendship, 1, 2)
        terms = [LayerEdges(), InterlayerDependence(1, 2)]
        fit = @test_logs (:warn, r"L\(friendship&advice\)~edges are at their smallest attainable values") match_mode=:any ergm_multi(
            m, terms; method=:mple)
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
        @test occursin("L(friendship&advice)~edges fixed at -Inf", out) && occursin("smallest attainable", out)
        @test any(occursin("fixed at -Inf", a) for a in approximations(fit))
        # `se=:bootstrap` is refused: nothing can be simulated at -Inf
        msg = errmsg(() -> ergm_multi(m, terms; method=:mple, se=:bootstrap, n_boot=5))
        @test_throws ArgumentError ergm_multi(m, terms; method=:mple, se=:bootstrap, n_boot=5)
        @test occursin("±Inf", msg) && occursin("L(friendship&advice)~edges", msg) && occursin("se=:hessian", msg)

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
        fs = @test_logs (:warn, r"L\(a\)~nodematch.grp are at their smallest attainable values") match_mode=:any ergm_multi(
            ms, [LayerEdges(), WithinLayer(NodeMatch(:grp), 1)])
        @test coef(fs)[2] == -Inf && stderror(fs)[2] == 0.0
        # 30 within-layer dyads, 6 within-group ones in layer 1 dropped; 7 ties
        # among the remaining 24
        @test coef(fs)[1] ≈ log(7 / 17) atol = 1e-8
        @test !is_exact(fs)

        # Separation by a COMBINATION of columns (ERGM.jl's `sep2` design lifted
        # into layer 1, with an unrelated layer 2): no single column is at its
        # boundary, yet nodecov.x + nodecov.z perfectly predicts layer 1's
        # ties. R ergm.multi warns "The MPLE does not exist!" (and returns
        # finite coefficients); here NetworkCore's separation verdict flags it,
        # names the two terms, and the fit follows the ecosystem's separation
        # policy: warn, converged = false, terms flagged, inference withheld.
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
        @test !has_dyad_dependent(smodel)        # `method=:auto` fits the MPLE
        Xf, y, η0 = ERGMMulti._multi_mple_design(sep, smodel.terms, smodel.offsets, [1, 2, 3, 4])
        @test isempty(ERGM.Extension.boundary_columns(Xf, ones(length(y)), Float64.(y)))
        sep_re = r"ergm_multi: the MPLE does not exist \(separation\)\. .*L\(a\)~nodecov\.x.*L\(a\)~nodecov\.z.*R ergm\.multi warns \"The MPLE does not exist!\""s
        fsep = @test_logs (:warn, sep_re) match_mode=:any ergm_multi(sep, sterms)
        @test !fsep.converged && fsep.separated
        @test fsep.separated_terms == ["L(a)~nodecov.x", "L(a)~nodecov.z"]
        @test !is_exact(fsep)
        @test all(isfinite, coef(fsep))         # the last iterate, returned but flagged
        # Inference withheld: NaN z, p and intervals on every coefficient
        @test all(isnan, coeftable(fsep).z_values) && all(isnan, coeftable(fsep).p_values)
        @test all(isnan, confint(fsep))
        @test all(isfinite, stderror(fsep)[1:2])  # kept, for diagnosis
        @test any(occursin("does not exist", a) && occursin("`L(a)~nodecov.x`", a)
                  for a in approximations(fsep))
        outs = sprint(show, fsep)
        @test occursin("converged: false", outs) && occursin("MPLE does not exist", outs) &&
              occursin("L(a)~nodecov.z", outs)
        # One warning: the separation sentence, not the generic one as well
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
        # ... and `se=:bootstrap` on a separated fit is refused, naming the terms
        msg = errmsg(() -> ergm_multi(sep, sterms; se=:bootstrap, n_boot=5))
        @test occursin("se=:bootstrap is not available on a separated fit", msg) &&
              occursin("L(a)~nodecov.x, L(a)~nodecov.z", msg)
        # ... and so is an MCMLE started from it
        msg = errmsg(() -> ergm_multi(sep, [sterms; WithinLayer(Kstar(2), 2)];
                                      n_samples=32))
        @test occursin("the MPLE start does not exist (separation", msg) &&
              occursin("L(a)~nodecov.x, L(a)~nodecov.z", msg)

        # The SAME separated design with a non-zero OFFSET is flagged too: a
        # finite offset shifts the linear predictor and cannot change the
        # verdict. (Before the shared verdict this case was skipped: the fit
        # came back `converged = true, separated = false` with coefficients
        # 20.7 / 25.1 and standard errors in the tens of thousands, and
        # `approximations` listed nothing but the offset note.)
        for offs in (Dict(2 => log(4 / 11)), Dict(1 => -1.0), Dict(2 => 0.7, 1 => -3.0))
            fso = @test_logs (:warn, sep_re) match_mode=:any ergm_multi(sep, sterms; offsets=offs)
            @test !fso.converged && fso.separated && !is_exact(fso)
            @test fso.separated_terms == ["L(a)~nodecov.x", "L(a)~nodecov.z"]
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
        @test isempty(fok.separated_terms)
        # and the golden offset fit (twolayer_ergm_multi.toml) stays unflagged
        # (asserted converged in its own testset)

        # The verdict is the shared NetworkCore one on the fitted design
        v = NetworkCore.logistic_separation(Xf, y)
        @test v.separated && v.certified && v.terms == [3, 4]
        Xk, yk, _ = ERGMMulti._multi_mple_design(fixture(), MultiERGMModel([LayerEdges(1), LayerEdges(2)], fixture()).terms,
                                                 Dict(2 => -2.0), [1])
        @test !NetworkCore.logistic_separation(Xk, yk).separated
        # No local detector is left
        src = _readtext(joinpath(@__DIR__, "..", "src", "ERGMMulti.jl"))
        @test !occursin("ERGM._separated", src) && !occursin("ERGM._warn_separated", src)
        @test !isdefined(ERGMMulti, :_multi_separated)

        # No "compatibility" constructors that default `se_type`,
        # `boot_replicates` or `separated`: the only way to build a result is
        # to pass all fifteen fields, so its flags cannot silently disagree
        # with the numbers it carries — and the separation flag and the
        # flagged terms are checked against each other
        @test fieldcount(MultiERGMResult) == 15
        @test all(mt -> mt.nargs - 1 == fieldcount(MultiERGMResult), methods(MultiERGMResult))
        @test_throws MethodError MultiERGMResult(fsep.model, coef(fsep), stderror(fsep), vcov(fsep),
                                                 loglikelihood(fsep), aic(fsep), bic(fsep), false, :hessian)
        @test_throws MethodError MultiERGMResult(fsep.model, coef(fsep), stderror(fsep), vcov(fsep),
                                                 loglikelihood(fsep), aic(fsep), bic(fsep), false, :hessian, nothing)
        rebuild(sep_flag, terms) = MultiERGMResult(fsep.model, coef(fsep), stderror(fsep),
            vcov(fsep), loglikelihood(fsep), aic(fsep), bic(fsep), false, :hessian, nothing,
            sep_flag, false, :mple, nothing, terms)
        @test rebuild(true, fsep.separated_terms).separated
        @test_throws ArgumentError rebuild(true, String[])
        @test_throws ArgumentError rebuild(false, ["L(a)~nodecov.x"])
    end

    @testset "A rank-deficient design: the duplicate is not identifiable (NaN, R's NA)" begin
        # Two copies of the same statistic: the second is a linear combination
        # of the first, so the pseudo-Hessian is singular. Newton used to stop
        # at its start (θ = 0 for both, NaN standard errors, "did not
        # converge"). As R's glm, the duplicate is reported as NaN (NA) with
        # a warning, and the first is fitted without it — logit(1/869), the
        # density of layer 1 (1 tie among 870 ordered dyads)
        m = MultilayerNetwork(30; directed=true)
        add_layer!(m, :friendship); add_layer!(m, :advice)
        add_layer_edge!(m, :friendship, 1, 2)
        add_layer_edge!(m, :advice, 2, 3)
        logs_d, fd = Test.collect_test_logs() do
            ergm_multi(m, [LayerEdges(1), LayerEdges(1)])
        end
        @test any(l -> occursin("are linear combinations of the preceding statistics",
                                string(l.message)), logs_d)
        @test !any(l -> occursin("did not converge", string(l.message)), logs_d)
        @test fd.converged && !fd.separated
        @test coef(fd)[1] ≈ log(1 / 869) atol = 1e-10
        @test isnan(coef(fd)[2]) && isnan(stderror(fd)[2]) && isfinite(stderror(fd)[1])
        @test dof(fd) == 1 && !is_exact(fd)
        @test any(occursin("not identifiable", a) for a in approximations(fd))
        @test occursin("not identifiable", sprint(show, fd))
        @test isnan(coeftable(fd)[2].p_value)
        @test occursin("not identifiable", errmsg(() -> ergm_multi(m, [LayerEdges(1), LayerEdges(1)];
                                                                    se=:bootstrap, n_boot=5)))
        # a statistic whose change statistics are all 0 and has no attainable
        # bound (an all-zero covariate): NaN with R's "not varying" sentence
        set_vertex_attribute!(m.layers[1], :z, Dict(v => 0.0 for v in 1:30))
        logs_z, fz = Test.collect_test_logs() do
            ergm_multi(m, [LayerEdges(1), WithinLayer(NodeCov(:z), 1)])
        end
        @test any(l -> occursin("do not vary on the dyads fitted", string(l.message)), logs_z)
        @test isnan(coef(fz)[2])
        @test coef(fz)[1] ≈ log(1 / 869) atol = 1e-10
        # `maxiter=1` on the golden design's terms: the generic sentence, once
        logs, f1 = Test.collect_test_logs() do
            ergm_multi(m, [LayerEdges(1), LayerEdges(2)]; maxiter=1)
        end
        @test !f1.converged && !is_exact(f1)
        @test count(l -> occursin("did not converge within maxiter = 1", string(l.message)), logs) == 1
        @test occursin("did not converge", join(approximations(f1)))
    end

    @testset "Bootstrap replicates without a finite MPLE are excluded" begin
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
        kw = (method=:mple, se=:bootstrap, n_boot=40, boot_burnin=200, boot_interval=20)
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
        @test NetworkCore.missing_policies(simulate_multi_ergm) == (:error,)
        @test NetworkCore.missing_policies(ergm_multi) == (:error,)
        @test !NetworkCore.supports_missing(simulate_multi_ergm)
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
        # Every coefficient label is R's, exactly (the layers are named A and
        # B, as in the R script's `Layer(list(A = ..., B = ...))`)
        @test smodel.terms.names == g.values["stat_names"]
        @test name(WithinLayer(GWESP(0.5), 1), m) == "L(A)~gwesp.OTP.fixed.0.5"

        # --- (ii) the dyad-dependent MPLE -----------------------------------
        terms = [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2), LayerMutual(1)]
        @test g.values["mple_julia_terms"] == ["LayerEdges(1)", "LayerEdges(2)",
                                               "InterlayerDependence(1, 2)", "LayerMutual(1)"]
        fit = ergm_multi(m, terms; method=:mple)
        @test fit.converged && !fit.separated
        @test has_dyad_dependent(fit.model) && !is_exact(fit)
        @test coeftable(fit).names == g.values["mple_term_names"]
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
        f = NetworkCore.logistic_derivatives(Xf, y; offset=η0)
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
        # The undirected labels are R's (no OTP on an undirected layer)
        @test umodel.terms.names == g.values["und_stat_names"]
        # The dyad-dependent MPLE over the UNORDERED within-layer dyads:
        # ergmMPLE's weights sum to 2 · 12 · 11 / 2 = 132, and so does nobs
        uterms = [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2), LayerTriangle(1)]
        @test g.values["und_mple_julia_terms"] == ["LayerEdges(1)", "LayerEdges(2)",
                                                   "InterlayerDependence(1, 2)", "LayerTriangle(1)"]
        ufit = ergm_multi(mu, uterms; method=:mple)
        @test ufit.converged && !ufit.separated && has_dyad_dependent(ufit.model)
        @test coeftable(ufit).names == g.values["und_mple_term_names"]
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
        fu = NetworkCore.logistic_derivatives(Xu, yu; offset=ηu)
        @test loglikelihood(ufit) >= fu(Float64.(g.values["und_exact_mple_coefficients"]))[1] - 1e-7
    end

    # ------------------------------------------------------------------
    # Sampler on the shared kernel, adapters under the conversion contract,
    # precompile workload, CI gates
    # ------------------------------------------------------------------
    @testset "Sampler is bit-identical to the pre-kernel loop (literal pins)" begin
        # The edge lists below were produced by a verbatim replica of the
        # committed pre-kernel loop (`simulate_multi_ergm` at
        # commit 0a82f9a "added bib citation": layer, i, j
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

    @testset "Conversion contract: combine_networks, split_by_layer, as_multilayer" begin
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

    # ------------------------------------------------------------------
    # Pooled within-layer terms, the MPLE presentation, auxiliary GOF panels
    # ------------------------------------------------------------------
    @testset "WithinLayer pools any ERGM.jl term across layers" begin
        # A random L-layer network with a two-level, a three-level and a
        # numeric attribute on every layer
        function random_multilayer(rng, n, L, directed; p=0.3)
            m = MultilayerNetwork(n; directed=directed)
            for l in 1:L
                add_layer!(m, Symbol("l", l))
                for i in 1:n, j in (directed ? (1:n) : (i+1:n))
                    i != j && rand(rng) < p && add_layer_edge!(m, l, i, j)
                end
                net = layer_network(m, l)
                for v in 1:n
                    set_vertex_attribute!(net, :grp, v, isodd(v) ? "a" : "b")
                    set_vertex_attribute!(net, :g3, v, ("a", "b", "c")[mod1(v, 3)])
                    set_vertex_attribute!(net, :x, v, round(sin(v); digits=3))
                end
            end
            return m
        end
        dir_terms = AbstractERGMTerm[Edges(), Mutual(), Triangle(), GWESP(0.5), GWDSP(0.5),
            GWNSP(0.5), OStar(2), IStar(2), TwoPath(), NodeMatch(:grp), NodeFactor(:g3),
            NodeMix(:grp), NodeCov(:x), AbsDiff(:x), IDegree(0:2), ODegree(1:2),
            GWIDegree(0.3), GWODegree(0.3), MeanDeg(), Density(), IDegRange(1, 3),
            ODegRange(2), Sender(nodes=[2, 3]), Receiver(nodes=[2, 3]),
            NodeMismatch(:grp)]
        und_terms = AbstractERGMTerm[Edges(), Triangle(), Kstar(2), GWESP(0.5), GWDSP(0.5),
            GWNSP(0.5), GWDegree(0.3), Degree(0:2), Concurrent(), NodeMatch(:grp),
            NodeFactor(:g3), NodeMix(:g3), NodeCov(:x), AbsDiff(:x), DegRange(2, 4),
            MeanDeg(), Density(), NodeMismatch(:grp)]
        rng = Xoshiro(20261002)
        n_checked = 0
        n_wrong = 0
        for directed in (true, false)
            m = random_multilayer(rng, 7, 3, directed)
            for t in (directed ? dir_terms : und_terms), sel in (2, [3, 1], :)
                w = WithinLayer(t, sel)
                layers = sel isa Colon ? (1:3) : sel
                # The statistic is the sum of the per-layer ERGM.jl statistics.
                # (Expanding terms are compared through the model below.)
                model = MultiERGMModel([w], m)
                per_layer = [collect(summary_stats(layer_network(m, l), [t])) for l in layers]
                @test collect(compute_all(model.terms, m)) ≈ sum(per_layer) atol=1e-10
                # The labels: the selector, then ERGM.jl's label on the layer
                # (`ergm.multi`'s: `L(l2)~…` for one layer, `L((l3,l1))~…` for
                # a pool over a list, every layer for `:`)
                lnames = String.(layer_names(m))
                lab = sel isa Int ? lnames[sel] : "(" * join(lnames[collect(layers)], ",") * ")"
                @test all(startswith("L($lab)~"), model.terms.names)
                # Change statistics against brute force, on the MATERIALIZED
                # terms the fit and the sampler use
                for mt in model.terms.terms, l in 1:3, i in 1:7, j in 1:7
                    (i == j || (!directed && i > j)) && continue
                    n_checked += 1
                    isapprox(change_stat_layer(mt, m, l, i, j), brute_change(mt, m, l, i, j);
                             atol=1e-9) || (n_wrong += 1)
                end
                # A raw (unmaterialized) single-statistic pooled term too
                if length(model.terms) == 1
                    @test compute(w, m) ≈ only(sum(per_layer)) atol=1e-10
                    @test change_stat_layer(w, m, first(layers), 1, 2) ≈
                          brute_change(w, m, first(layers), 1, 2) atol=1e-9
                end
            end
        end
        @test n_checked > 10_000
        @test n_wrong == 0

        # The dedicated pooled terms are the same statistics
        m = fixture()
        @test compute(WithinLayer(Edges(), :), m) == compute(LayerEdges(), m)
        @test compute(WithinLayer(Mutual(), [1, 2]), m) == compute(LayerMutual([1, 2]), m)
        @test compute(WithinLayer(Triangle(), [1, 2]), m) == compute(LayerTriangle([1, 2]), m)
        @test WithinLayer(Edges(), 1:2).layer == [1, 2]          # a range is a vector
        @test name(WithinLayer(GWESP(0.5), [1, 2]), m) == "L((friendship,advice))~gwesp.OTP.fixed.0.5"
        @test name(WithinLayer(GWESP(0.5), :), fixture_undirected()) ==
              "L((friendship,advice))~gwesp.fixed.0.5"
        @test name(WithinLayer(Edges(), [1, 2])) == "L((1,2))~edges"
        # Dyad dependence follows the wrapped term; a pooled fit is one coefficient
        @test !has_dyad_dependent(MultiERGMModel([WithinLayer(Edges(), :)], m))
        @test has_dyad_dependent(MultiERGMModel([WithinLayer(GWESP(0.5), [1, 2])], m))
        fit_pool = ergm_multi(m, [WithinLayer(Edges(), :)])
        fit_layer = ergm_multi(m, [LayerEdges()])
        @test coef(fit_pool) ≈ coef(fit_layer) atol=1e-12
        @test stderror(fit_pool) ≈ stderror(fit_layer) atol=1e-12

        # Refusals, in words
        @test occursin("offsets=", errmsg(() -> WithinLayer(Offset(Edges(), -1.0), 1)))
        @test occursin("repeats a layer", errmsg(() -> WithinLayer(Edges(), [1, 1])))
        @test occursin("selects layers by index", errmsg(() -> WithinLayer(Edges(), [:a, :b])))
        @test occursin("selects no layer",
                       errmsg(() -> MultiERGMModel([WithinLayer(Edges(), Int[])], m)))
        msg = errmsg(() -> MultiERGMModel([WithinLayer(Edges(), [1, 3])], m))
        @test occursin("layer 3", msg) && occursin(":advice", msg)
        # Validation runs on EVERY selected layer and names the one that fails
        m2 = fixture()
        set_vertex_attribute!(layer_network(m2, 1), :grp, Dict(v => "a" for v in 1:4))
        msg = errmsg(() -> MultiERGMModel([WithinLayer(NodeMatch(:grp), [1, 2])], m2))
        @test occursin("layer 2 (:advice)", msg) && occursin("grp", msg)
        msg = errmsg(() -> MultiERGMModel([WithinLayer(Kstar(2), :)], m))
        @test occursin("layer 1 (:friendship)", msg) && occursin("OStar(2)", msg)
        # An expanding pooled term needs the same levels on every layer
        set_vertex_attribute!(layer_network(m2, 1), :grp, Dict(1 => "a", 2 => "b", 3 => "a", 4 => "b"))
        set_vertex_attribute!(layer_network(m2, 2), :grp, Dict(1 => "a", 2 => "b", 3 => "c", 4 => "b"))
        msg = errmsg(() -> MultiERGMModel([WithinLayer(NodeFactor(:grp), [1, 2])], m2))
        @test occursin("expands into different statistics", msg) &&
              occursin("layer 2: nodefactor.grp.b, nodefactor.grp.c", msg)
        # Offsets on a pooled term: by index, as for every multilayer term
        fit_off = ergm_multi(m, [LayerEdges(), WithinLayer(Mutual(), :)]; method=:mple, offsets=Dict(2 => 0.5))
        @test coef(fit_off)[2] == 0.5 && isnan(stderror(fit_off)[2])

        # The change-statistic fill of a pooled attribute term is allocation-free
        big = random_multilayer(Xoshiro(7), 30, 2, true)
        model = MultiERGMModel([LayerEdges(), WithinLayer(NodeMatch(:grp), :),
                                WithinLayer(GWESP(0.5), [1, 2]), WithinLayer(NodeFactor(:g3), :)], big)
        fill_all(dest, ts, m) = (for l in 1:2, i in 1:30, j in 1:30
                                     i == j || ERGMMulti._change_stat_layer_all!(dest, ts, m, l, i, j)
                                 end; dest)
        dest = zeros(length(model.terms))
        ts = model.terms
        fill_all(dest, ts, big)
        @test (@allocated fill_all(dest, ts, big)) == 0
        # ... and it simulates: the pooled statistic's mean tracks its coefficient
        lo = simulate_multi_ergm(big, [WithinLayer(Edges(), :)], [-2.0]; n_sim=20, rng=Xoshiro(1))
        hi = simulate_multi_ergm(big, [WithinLayer(Edges(), :)], [0.0]; n_sim=20, rng=Xoshiro(1))
        @test mean(compute(LayerEdges(), s) for s in lo) < mean(compute(LayerEdges(), s) for s in hi)
    end

    @testset "Golden fixture: ergm.multi pooled layer terms L(~t, c(~A, ~B))" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "pooled_layer_terms.toml"))
        g1 = load_golden(joinpath(@__DIR__, "fixtures", "twolayer_layer_terms.toml"))
        @test g.provenance["ergm_multi_version"] == "0.3.0"
        @test g.provenance["r_warnings"] == "none (stopifnot in the script)"
        # One network, three fixtures
        for key in ("layer_src", "layer_dst", "und_layer_src", "und_layer_dst")
            @test g.values[key] == g1.values[key]
        end
        function build(n, directed, src_key, dst_key, grp, g3, x)
            local m = MultilayerNetwork(n; directed=directed)
            for l in 1:2
                net = network(n; directed=directed)
                for v in 1:n
                    set_vertex_attribute!(net, :grp, v, grp[v])
                    set_vertex_attribute!(net, :g3, v, g3[v])
                    set_vertex_attribute!(net, :x, v, x[v])
                end
                s, d = Int.(g.values[src_key][l]), Int.(g.values[dst_key][l])
                for k in eachindex(s)
                    add_edge!(net, s[k], d[k])
                end
                add_layer!(m, l == 1 ? :A : :B; net=net)
            end
            return m
        end
        m = build(Int(g.values["n_actors"]), true, "layer_src", "layer_dst",
                  String.(g.values["grp"]), String.(g.values["g3"]), Float64.(g.values["x"]))
        mu = build(Int(g.values["und_n_actors"]), false, "und_layer_src", "und_layer_dst",
                   String.(g.values["und_grp"]), String.(g.values["und_g3"]),
                   Float64.(g.values["und_x"]))
        term(str) = fixture_term(str)

        for (net, pre) in ((m, ""), (mu, "und_"))
            # (i)/(iii) the statistics: every term pooled over both layers,
            # R's label for every statistic, R's value at 1e-10
            terms = [WithinLayer(term(t), [1, 2]) for t in g.values[pre * "stat_julia_terms"]]
            model = MultiERGMModel(terms, net)
            @test model.terms.names == g.values[pre * "stat_names"]
            vals = collect(compute_all(model.terms, net))
            @test check_golden(g, pre * "statistics", vals) ||
                  error(golden_report(g, pre * "statistics", vals))
            # `:` is the same pool on a two-layer network
            model_all = MultiERGMModel([WithinLayer(term(t), :)
                                        for t in g.values[pre * "stat_julia_terms"]], net)
            @test check_golden(g, pre * "statistics", collect(compute_all(model_all.terms, net)))

            # (ii)/(iv) the dyad-dependent MPLE with pooled terms
            fterms = AbstractERGMTerm[term(t) for t in g.values[pre * "mple_julia_terms"]]
            fit = ergm_multi(net, fterms; method=:mple)
            @test fit.converged && !fit.separated && has_dyad_dependent(fit.model)
            @test fit.model.terms.names == g.values[pre * "mple_term_names"]
            @test nobs(fit) == Int(g.values[pre * "mple_dyad_universe"])
            for key in ("mple_coefficients", "exact_mple_coefficients")
                @test check_golden(g, pre * key, fit.coefficients) ||
                      error(golden_report(g, pre * key, fit.coefficients))
            end
            for key in ("mple_std_errors", "exact_mple_std_errors")
                @test check_golden(g, pre * key, fit.std_errors) ||
                      error(golden_report(g, pre * key, fit.std_errors))
            end
        end
    end

    @testset "Golden fixture: ergm.multi coefficient labels" begin
        # The labels R prints for every multilayer term — one layer, a pool
        # over a subset, over every layer and over one layer, both orders of a
        # layer pair, mutualL, attribute and expanding terms inside L(), and
        # offset terms — compared EXACTLY, on three named directed layers and
        # two undirected ones; the statistics they label at 1e-10.
        g = load_golden(joinpath(@__DIR__, "fixtures", "multilayer_labels.toml"))
        @test g.provenance["ergm_multi_version"] == "0.3.0"
        n = Int(g.values["n_actors"])
        gv, xv = String.(g.values["g"]), Float64.(g.values["x"])
        function build(directed, names, keys)
            local m = MultilayerNetwork(n; directed=directed)
            for (lname, key) in zip(names, keys)
                net = network(n; directed=directed)
                for v in 1:n
                    set_vertex_attribute!(net, :g, v, gv[v])
                    set_vertex_attribute!(net, :x, v, xv[v])
                end
                for e in g.values[key]
                    add_edge!(net, Int(e[1]), Int(e[2]))
                end
                add_layer!(m, Symbol(lname); net=net)
            end
            return m
        end
        term(str) = fixture_term(str)
        # Only term constructors applied to literals are evaluated
        @test fixture_term("WithinLayer(IDegree(1:2), [1, 3])") isa WithinLayer
        @test_throws ErrorException fixture_term("run(`true`)")
        @test_throws ErrorException fixture_term("LayerEdges(begin; x = 1; end)")
        @test_throws ErrorException fixture_term("Base.rm(\"x\")")
        m3 = build(true, g.values["layer_names"],
                   ["edges_friend", "edges_advice", "edges_cowork"])
        mu = build(false, g.values["und_layer_names"], ["und_edges_a", "und_edges_b"])
        @test layer_names(m3) == [:friend, :advice, :cowork]
        for (net, pre) in ((m3, ""), (mu, "und_"))
            model = MultiERGMModel([term(t) for t in g.values[pre * "stat_julia_terms"]], net)
            @test model.terms.names == g.values[pre * "stat_names"]
            vals = collect(compute_all(model.terms, net))
            @test check_golden(g, pre * "statistics", vals) ||
                  error(golden_report(g, pre * "statistics", vals))
        end
        # Offset terms are labelled `offset(<label>)`, as R labels them
        oterms = [term(t) for t in g.values["offset_julia_terms"]]
        ov = Float64.(g.values["offset_values"])
        ofit = ergm_multi(m3, oterms; offsets=Dict(3 => ov[1], 4 => ov[2]), method=:mple)
        @test coeftable(ofit).names == g.values["offset_coef_names"]
        @test occursin("offset(L(friend&advice)~edges)", sprint(show, ofit))
    end

    @testset "MPLE of a dyad-dependent formula withholds naive inference by default" begin
        rng = Xoshiro(11)
        m = MultilayerNetwork(12; directed=true)
        add_layer!(m, :a); add_layer!(m, :b)
        for l in 1:2, i in 1:12, j in 1:12
            i != j && rand(rng) < 0.25 && add_layer_edge!(m, l, i, j)
        end
        dep = [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2), LayerMutual(1)]
        ind = [LayerEdges(1), LayerEdges(2)]

        # Default: estimates and naive SEs, no z / p / interval
        fit = ergm_multi(m, dep; method=:mple)
        @test fit.inference_withheld && se_method(fit) === :hessian
        @test all(isfinite, coef(fit)) && all(isfinite, stderror(fit))
        tbl = coeftable(fit)
        @test all(isnan, tbl.z_values) && all(isnan, tbl.p_values)
        msg = errmsg(() -> confint(fit))
        @test occursin("se=:bootstrap", msg) && occursin("se=:hessian", msg) &&
              occursin("under-cover", msg)
        @test_throws ArgumentError confint(fit; level=0.9)
        out = sprint(show, fit)
        @test occursin("z values and p-values are not reported (NaN)", out)
        @test occursin("se=:bootstrap", out) && occursin("se=:hessian", out)
        @test occursin("biased in finite samples", out)
        @test any(occursin("withheld", a) for a in approximations(fit))
        @test !is_exact(fit)

        # Explicit se=:hessian: the same numbers, R's naive Wald table, the warning
        naive = ergm_multi(m, dep; method=:mple, se=:hessian)
        @test !naive.inference_withheld
        @test coef(naive) == coef(fit) && stderror(naive) == stderror(fit)
        tn = coeftable(naive)
        @test tn.z_values ≈ coef(naive) ./ stderror(naive)
        @test all(p -> 0 <= p <= 1, tn.p_values)
        @test size(confint(naive)) == (4, 2)
        outn = sprint(show, naive)
        @test occursin("Warning: this model contains dyad-dependent terms", outn)
        @test !any(occursin("withheld", a) for a in approximations(naive))
        @test any(occursin("anticonservative", a) for a in approximations(naive))

        # se=:bootstrap: calibrated table; the point-estimate caveat stays
        boot = ergm_multi(m, dep; method=:mple, se=:bootstrap, n_boot=40, rng=Xoshiro(3))
        @test !boot.inference_withheld
        @test all(isfinite, coeftable(boot).p_values)
        @test size(confint(boot)) == (4, 2)
        outb = sprint(show, boot)
        @test occursin("point estimates are biased", outb)
        @test any(occursin("biased in finite samples", a) for a in approximations(boot))

        # A dyad-independent formula is exact: nothing is withheld
        exact = ergm_multi(m, ind)
        @test !exact.inference_withheld && is_exact(exact)
        @test all(isfinite, coeftable(exact).p_values)
        @test size(confint(exact)) == (2, 2)
        @test !occursin("not reported", sprint(show, exact))

        # A coefficient fixed at -Inf keeps its conventional z = -Inf, p = 0
        mb = MultilayerNetwork(6; directed=true)
        add_layer!(mb, :a); add_layer!(mb, :b)
        for (i, j) in ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1))
            add_layer_edge!(mb, :a, i, j); add_layer_edge!(mb, :b, j, i)
        end
        fb = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            ergm_multi(mb, [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2)]; method=:mple)
        end
        tb = coeftable(fb)
        @test fb.inference_withheld
        @test tb.z_values[3] == -Inf && tb.p_values[3] == 0.0
        @test isnan(tb.z_values[1]) && isnan(tb.p_values[1])

        # The keyword's vocabulary is unchanged
        @test occursin(":sandwich", errmsg(() -> ergm_multi(m, ind; se=:sandwich)))
    end

    @testset "gof: auxiliary panels (multiplexity, degree, ESP per layer)" begin
        # Definitions against brute force on the fixtures
        for (m, directed) in ((fixture(), true), (fixture_undirected(), false))
            n, L = m.n, 2
            A = [Bool[has_edge(layer_network(m, l), i, j) for i in 1:n, j in 1:n] for l in 1:L]
            mult = zeros(L + 1)
            for i in 1:n, j in 1:n
                (i == j || (!directed && i > j)) && continue
                mult[sum(A[l][i, j] for l in 1:L) + 1] += 1
            end
            @test ERGMMulti._multiplexity_counts(m) == mult
            @test sum(mult) == (directed ? n * (n - 1) : n * (n - 1) ÷ 2)
            for l in 1:L
                net = layer_network(m, l)
                od = [count(A[l][v, :]) for v in 1:n]
                id = [count(A[l][:, v]) for v in 1:n]
                cnt(d) = [Float64(count(==(k), d)) for k in 0:(n - 1)]
                if directed
                    @test ERGMMulti._degree_counts(net, :out) == cnt(od)
                    @test ERGMMulti._degree_counts(net, :in) == cnt(id)
                else
                    @test ERGMMulti._degree_counts(net, :total) == cnt(od)
                end
                esp = zeros(n - 1)
                for i in 1:n, j in 1:n
                    (A[l][i, j] && (directed || i < j)) || continue
                    sp = count(k -> k != i && k != j && A[l][i, k] && A[l][k, j], 1:n)
                    esp[sp + 1] += 1
                end
                @test ERGMMulti._esp_counts(net) == esp
                @test sum(esp) == ne(net)
                # The ESP distribution is the one GWESP is built on
                @test sum(esp[k + 1] * (ℯ^0.5 * (1 - (1 - ℯ^-0.5)^k)) for k in 0:(n - 2)) ≈
                      compute(GWESP(0.5), net) atol=1e-10
            end
        end

        # The panels of a fit
        rng = Xoshiro(21)
        m = MultilayerNetwork(10; directed=true)
        add_layer!(m, :a); add_layer!(m, :b)
        for l in 1:2, i in 1:10, j in 1:10
            i != j && rand(rng) < 0.3 && add_layer_edge!(m, l, i, j)
        end
        g = gof(ergm_multi(m, [LayerEdges(1), LayerEdges(2)]); n_sim=60, rng=Xoshiro(22))
        @test [p.name for p in g.statistics] ==
              ["model statistics", "layer edges", "multiplexity", "idegree: a", "odegree: a",
               "esp: a", "idegree: b", "odegree: b", "esp: b"]
        mp = g.statistics[3]
        @test mp.labels == ["0", "1", "2"] && sum(mp.observed) == 90
        @test all(sum(mp.simulated; dims=2) .== 90)
        for p in g.statistics[4:end]
            @test p.labels == string.(0:(length(p.labels) - 1))
            total = startswith(p.name, "esp") ? ne(layer_network(m, Symbol(p.name[end]))) : 10
            @test sum(p.observed) == total
            @test all(0 .< p.p_values .<= 1)
        end
        # Undirected layers get one degree panel each
        mu = fixture_undirected()
        gu = gof(ergm_multi(mu, [LayerEdges()]); n_sim=30, rng=Xoshiro(23))
        @test [p.name for p in gu.statistics] ==
              ["model statistics", "layer edges", "multiplexity", "degree: friendship",
               "esp: friendship", "degree: advice", "esp: advice"]

        # Power: data with strong cross-layer overlap, fitted WITHOUT the
        # cross-layer term — the model statistics fit by construction, the
        # multiplexity panel flags the misfit
        rng = Xoshiro(31)
        mo = MultilayerNetwork(14; directed=true)
        add_layer!(mo, :a); add_layer!(mo, :b)
        for i in 1:14, j in 1:14
            i != j && rand(rng) < 0.3 && (add_layer_edge!(mo, :a, i, j); add_layer_edge!(mo, :b, i, j))
        end
        go = gof(ergm_multi(mo, [LayerEdges(1), LayerEdges(2)]); n_sim=99, rng=Xoshiro(32))
        @test all(go.statistics[1].p_values .> 0.2)
        @test go.statistics[3].p_values[end] <= 0.02
    end

    @testset "Excluded bootstrap replicates are disclosed as a downward bias" begin
        # The ERGM family's one sentence, verbatim, in the warning, `show` and
        # `approximations`
        sentence = "The standard errors are conditional on a finite refit: the " *
                   "excluded replicates are the extreme ones, so the standard " *
                   "errors are biased downward."
        @test ERGMMulti._BOOT_EXCLUSION_BIAS == sentence == ERGM._BOOT_EXCLUSION_BIAS
        m = fixture_undirected()
        terms = [LayerEdges(), InterlayerDependence(1, 2)]
        kw = (method=:mple, se=:bootstrap, n_boot=60)
        logs, fit = Test.collect_test_logs() do
            ergm_multi(m, terms; kw..., rng=Xoshiro(5))
        end
        n_bad = count(b -> !all(isfinite, view(fit.boot_replicates, b, :)), 1:60)
        @test n_bad > 0                       # the tiny fixture does produce them
        warned = [string(l.message) for l in logs if occursin("bootstrap refits", string(l.message))]
        @test length(warned) == 1 && occursin(sentence, only(warned))
        note = ERGMMulti._boot_exclusion_note(fit)
        @test occursin("$n_bad of the 60 bootstrap refits", note)
        @test occursin(sentence, note)
        @test note in approximations(fit)
        # `show` prints the note as one line
        @test occursin(sentence, sprint(show, fit))
    end

    # ------------------------------------------------------------------
    # MCMLE — ergm.multi's default estimator for a dyad-dependent model.
    # ------------------------------------------------------------------
    @testset "MCMLE: exact against enumeration on a 3-actor, two-layer network" begin
        # 2 layers × 6 ordered dyads = 12 dyads: 4096 states, so the
        # likelihood, its maximizer and the normalizing constant are sums
        dyads = [(i, j) for i in 1:3 for j in 1:3 if i != j]
        function build(bits)
            m = MultilayerNetwork(3; directed=true)
            add_layer!(m, :a); add_layer!(m, :b)
            for l in 1:2, (k, (i, j)) in enumerate(dyads)
                bits[(l - 1) * 6 + k] == 1 && add_layer_edge!(m, l, i, j)
            end
            return m
        end
        terms = [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2), LayerMutual(1)]
        obs = build([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
        model = MultiERGMModel(terms, obs)
        G = Matrix{Float64}(undef, 4096, 4)
        for s in 0:4095
            G[s + 1, :] = compute_all(model.terms, build(digits(s, base=2, pad=12)))
        end
        g_obs = compute_all(model.terms, obs)
        function exact(θ, cols=1:4, fixed=zeros(4))
            full = copy(fixed); full[cols] = θ
            η = G * full
            mx = maximum(η); w = exp.(η .- mx); Z = sum(w); w ./= Z
            μ = G' * w
            Σ = G' * (w .* G) - μ * μ'
            return (sum(full .* g_obs) - (mx + log(Z)), (g_obs .- μ)[cols], -Σ[cols, cols])
        end
        mle = NetworkCore.newton_fit(exact, zeros(4))
        @test mle.converged

        fit = ergm_multi(obs, terms; method=:mcmle, n_samples=4000, rng=Xoshiro(1))
        @test fit.method === :mcmle && fit.converged && fit.se_type === :mcmc
        mc = fit.mcmc
        @test mc.convergence isa ERGM.MCMLEConvergence
        @test all(abs.(coef(fit) .- mle.θ) .<= 5 .* mc.mc_std_errors)
        @test maximum(abs.(coef(fit) .- mle.θ) ./ mle.se) < 0.2
        @test all(isapprox.(stderror(fit), mle.se; rtol=0.1))
        @test abs(loglikelihood(fit) - mle.loglik) <= 5 * mc.loglik_mc_se + 0.03
        @test mc.loglik_mc_se < 0.1
        @test aic(fit) ≈ -2 * loglikelihood(fit) + 2 * dof(fit)
        @test bic(fit) ≈ -2 * loglikelihood(fit) + dof(fit) * log(nobs(fit))
        # the MPLE it started from is a different number on this network
        mple = ergm_multi(obs, terms; method=:mple)
        @test mc.start == coef(mple)
        @test maximum(abs.(coef(mple) .- mle.θ)) > 10 * maximum(mc.mc_std_errors)

        # The record and the protocol
        @test objective(fit) === :likelihood && objective(mple) === :pseudolikelihood
        @test se_method(fit) === :fisher && se_method(mple) === :hessian
        @test !is_exact(fit) && !fit.inference_withheld && !fit.separated
        @test fit.boot_replicates === nothing && mple.mcmc === nothing && mple.method === :mple
        @test size(confint(fit)) == (4, 2) && all(isfinite, coeftable(fit).z_values)
        @test NetworkCore.check_statsapi(fit; strict=true) !== nothing
        @test any(occursin("MCMLE: the likelihood is approximated", a) for a in approximations(fit))
        @test !any(occursin("pseudo-likelihood", a) for a in approximations(fit))
        out = sprint(show, fit)
        @test occursin("Method: mcmle (Monte-Carlo maximum likelihood)", out)
        @test occursin("log-likelihood:", out) && !occursin("pseudo", out)
        @test occursin("inverse Fisher information + Monte-Carlo error", out)
        # reproducible from `rng` alone
        again = ergm_multi(obs, terms; method=:mcmle, n_samples=4000, rng=Xoshiro(1))
        @test coef(again) == coef(fit) && loglikelihood(again) == loglikelihood(fit)
        @test coef(ergm_multi(obs, terms; method=:mcmle, n_samples=4000, rng=Xoshiro(2))) != coef(fit)
        # ... and from a prebuilt model; gof and simulation run from the fit
        @test coef(ergm_multi(model; method=:mcmle, n_samples=4000, rng=Xoshiro(1))) == coef(fit)
        gf = gof(fit; n_sim=200, rng=Xoshiro(4))
        @test all(p -> p > 0.05, gf.statistics[1].p_values)   # the MLE matches its statistics

        # Offsets stay fixed and enter every chain: with one coefficient pinned
        # at an arbitrary value, the others are the constrained exact MLE
        cmle = NetworkCore.newton_fit(θ -> exact(θ, [1, 2, 4], [0.0, 0.0, 0.5, 0.0]), zeros(3))
        off = ergm_multi(obs, terms; offsets=Dict(3 => 0.5), method=:mcmle,
                         n_samples=4000, rng=Xoshiro(5))
        @test off.converged && coef(off)[3] == 0.5 && isnan(stderror(off)[3])
        @test all(abs.(coef(off)[[1, 2, 4]] .- cmle.θ) .<= 5 .* off.mcmc.mc_std_errors)
        @test abs(loglikelihood(off) - cmle.loglik) <= 5 * off.mcmc.loglik_mc_se + 0.03
        @test dof(off) == 3

        # A dyad-INDEPENDENT formula needs no Monte Carlo: the exact MPLE
        di = ergm_multi(obs, [LayerEdges(1), LayerEdges(2)]; method=:mcmle)
        dm = ergm_multi(obs, [LayerEdges(1), LayerEdges(2)])
        @test di.method === :mcmle && di.mcmc === nothing && is_exact(di)
        @test coef(di) == coef(dm) && loglikelihood(di) == loglikelihood(dm)
        @test objective(di) === :likelihood
        @test occursin("exact: the formula is dyad-independent", sprint(show, di))

        # Refusals, in words
        msg = errmsg(() -> ergm_multi(obs, terms; method=:mcmc))
        @test occursin("unknown estimation method :mcmc", msg) && occursin(":mple, :mcmle", msg)
        @test occursin("keyword `se` is not accepted by method=:mcmle",
                       errmsg(() -> ergm_multi(obs, terms; method=:mcmle, se=:bootstrap)))
        @test occursin("n_samples must be at least 16",
                       errmsg(() -> ergm_multi(obs, terms; method=:mcmle, n_samples=4)))
        # no finite MLE where the statistic is at its boundary (no co-occurring
        # tie): fixed at -Inf (R's drop), or refused under drop=false
        nodup = build([1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        @test compute(InterlayerDependence(1, 2), nodup) == 0
        nd = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            ergm_multi(nodup, [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2)];
                       method=:mcmle, rng=Xoshiro(2))
        end
        @test coef(nd)[3] == -Inf && all(isfinite, coef(nd)[1:2])
        @test occursin("drop=false",
                       errmsg(() -> ergm_multi(nodup, [LayerEdges(1), LayerEdges(2),
                                                       InterlayerDependence(1, 2)];
                                               method=:mcmle, drop=false)))
        # a masked layer is refused, as for the MPLE
        masked = build([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
        set_missing_dyad!(masked.layers[1], 1, 2)
        @test_throws ArgumentError ergm_multi(masked, terms; method=:mcmle)
        # a fit that cannot converge in its budget says so
        unc = @test_logs (:warn, r"MCMLE did not converge in 1 iterations") match_mode = :any ergm_multi(
            obs, terms; method=:mcmle, n_samples=64, mcmle_maxiter=1,
            termination=:hotelling, conv_threshold=1e-9, bridge_rungs=0,
            rng=Xoshiro(1))
        @test !unc.converged && isnan(loglikelihood(unc))
        @test any(occursin("MCMLE did not converge", a) for a in approximations(unc))
        @test occursin("MCMLE did not converge", sprint(show, unc))
        # Under :hotelling the verdict is that rule: its p-value and the max
        # t-ratio are part of it
        @test any(occursin("Hotelling p", a) && occursin("max t-ratio", a) &&
                  occursin("step length γ", a) for a in approximations(unc))
        # MCMLE non-convergence is loud, and quotes the STOPPING RULE: under
        # the default confidence rule its equivalence-test p-value and the step
        # length γ — never the classical t-ratio / Hotelling diagnostics, which
        # describe the sample drawn before the last step and are not the rule
        logs, cunc = Test.collect_test_logs() do
            ergm_multi(obs, terms; method=:mcmle, n_samples=64, mcmle_maxiter=1,
                       conv_precision=1e-6, bridge_rungs=0, rng=Xoshiro(1))
        end
        @test !cunc.converged && cunc.mcmc.termination === :confidence
        @test cunc.mcmc.conv_confidence == 0.99 && cunc.mcmc.conv_precision == 1e-6
        warned = [string(l.message) for l in logs if occursin("MCMLE did not converge", string(l.message))]
        caveat = ERGMMulti._nonconvergence_caveat(cunc)
        printed = sprint(show, cunc)
        @test length(warned) == 1
        for text in (only(warned), caveat, printed)
            @test occursin("99% equivalence test p", text) && occursin("step length γ", text)
            @test !occursin("t-ratio", text) && !occursin("Hotelling", text)
        end
        @test caveat in approximations(cunc)

        # The chain behind it carries the statistics exactly: after any run
        # they equal `compute_all` on the state it ended in
        cur = build([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
        Gs = Matrix{Float64}(undef, 50, 4)
        ERGMMulti._multi_mh_stats!(Xoshiro(3), cur, model.terms, [-0.5, -0.5, 0.5, 0.5], Gs, 100, 7)
        @test Gs[end, :] == compute_all(model.terms, cur)
    end

    @testset "MCMLE is thread-count independent (fresh process)" begin
        m = fixture()
        terms = [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2)]
        fit = ergm_multi(m, terms; method=:mcmle, n_samples=256, rng=Xoshiro(7))
        other_threads = Threads.nthreads() == 1 ? 4 : 1
        script = """
            using ERGMMulti, Random
            m = MultilayerNetwork(4; directed=true)
            add_layer!(m, :friendship); add_layer!(m, :advice)
            for (i, j) in [(1, 2), (2, 1), (1, 3), (3, 4)]; add_layer_edge!(m, :friendship, i, j); end
            for (i, j) in [(1, 2), (2, 3), (3, 4), (4, 3)]; add_layer_edge!(m, :advice, i, j); end
            f = ergm_multi(m, [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2)];
                           method=:mcmle, n_samples=256, rng=Xoshiro(7))
            println(Threads.nthreads()); println(repr(coef(f))); println(repr(loglikelihood(f)))
            """
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$other_threads --project=$(dirname(@__DIR__)) -e $script`
        res = run_fresh(cmd; expect_lines=3)
        @test res.ok
        lines = split(strip(res.out), '\n')
        @test length(lines) == 3
        @test parse(Int, lines[1]) == other_threads
        @test lines[2] == repr(coef(fit))
        @test lines[3] == repr(loglikelihood(fit)) && isfinite(loglikelihood(fit))
    end

    @testset "Golden fixture: ergm.multi MCMLE of a dyad-dependent two-layer model" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "multilayer_mcmle.toml"))
        @test g.provenance["ergm_multi_version"] == "0.3.0"
        v(key) = Float64.(g.values[key])
        m = MultilayerNetwork(Int(g.values["n_actors"]); directed=true)
        add_layer!(m, :A); add_layer!(m, :B)
        for l in 1:2, (i, j) in zip(g.values["layer$(l)_src"], g.values["layer$(l)_dst"])
            add_layer_edge!(m, l, Int(i), Int(j))
        end
        terms = [fixture_term(t) for t in g.values["julia_terms"]]
        model = MultiERGMModel(terms, m)
        @test model.terms.names == g.values["term_names"]          # R's labels
        @test compute_all(model.terms, m) ≈ v("summary_statistics") atol = g.tolerance["summary_statistics"]
        n_seeds = length(g.values["seeds"])
        Rm, Rsd, Sm, Ssd = v("coefficients_mean"), v("coefficients_sd"),
                           v("std_errors_mean"), v("std_errors_sd")
        band = max.(g.tolerance["band_sd"] .* Rsd .* sqrt(1 + 1 / n_seeds),
                    g.tolerance["floor_se_fraction"] .* Sm)
        se_band = max.(g.tolerance["band_sd"] .* Ssd .* sqrt(1 + 1 / n_seeds),
                       g.tolerance["floor_se_fraction"] .* Sm)
        ll_band = max(g.tolerance["band_sd"] * sqrt(2) * g.values["loglik_sd"],
                      g.tolerance["loglik_floor"])
        for seed in (1, 2, 3)
            # `method=:auto` (the default) fits the MCMLE of this dyad-dependent model
            f = ergm_multi(model; rng=Xoshiro(seed))
            @test f.converged && f.method === :mcmle
            @test all(abs.(coef(f) .- Rm) .<= band)
            @test all(abs.(stderror(f) .- Sm) .<= se_band)
            @test abs(loglikelihood(f) - g.values["loglik_mean"]) <= ll_band
            # R's long-chain fit lies inside the same band
            @test all(abs.(coef(f) .- v("hp_coefficients")) .<= band)
            # the Monte-Carlo error is a few % of an SE. A stochastic pin:
            # under ERGM's shared confidence rule (precision 0.1) eight seeds
            # measured 0.047-0.072 (final samples of 3 992-8 984 draws); R's
            # own seed-to-seed sd is 3-5 % of its SE on this model
            @test maximum(f.mcmc.mc_std_errors ./ stderror(f)) <= 0.10
            @test f.mcmc.termination === :confidence && f.mcmc.termination_p < 0.01
        end
        # The MPLE of the same model is R's MPLE — as shipped at the fixture's
        # tolerance (20x R's measured glm slack) and refit exactly at 1e-6 —
        # and a different number: more than half a standard error from the
        # MLE on the gwesp coefficient
        mple = ergm_multi(model; method=:mple)
        @test check_golden(g, "mple_coefficients", coef(mple)) ||
              error(golden_report(g, "mple_coefficients", coef(mple)))
        @test check_golden(g, "exact_mple_coefficients", coef(mple)) ||
              error(golden_report(g, "exact_mple_coefficients", coef(mple)))
        @test abs(coef(mple)[5] - Rm[5]) / Sm[5] > 0.5
    end

    @testset "Not implemented (README, docs index) = Known limitations (CHANGELOG)" begin
        root = dirname(@__DIR__)
        # The bullet list from its first item to the next heading
        function limitations(path)
            txt = _readtext(joinpath(root, path))
            start = findfirst("- **The MCMLE is not R's implementation", txt)
            start === nothing && return ""
            rest = txt[first(start):end]
            stop = findfirst(r"\n## "m, rest)
            return strip(stop === nothing ? rest : rest[1:first(stop)])
        end
        readme = limitations("README.md")
        @test count("\n- **", "\n" * readme) == 5
        @test limitations(joinpath("docs", "src", "index.md")) == readme
        @test limitations("CHANGELOG.md") == readme
        for needle in ("method = :mcmle", "Networks()", "gofN", "L(~term, c(~A, ~B))",
                       "Layer(nw, c(\"attr1\", \"attr2\"))", ".symmetric", "geodesic")
            @test occursin(needle, readme)
        end
        # Release notes, not a work log
        changelog = _readtext(joinpath(root, "CHANGELOG.md"))
        @test !occursin(r"expert-panel|panel 20|panel review|round \d|item \d"i, changelog)
        @test !occursin("docs-stable", _readtext(joinpath(root, "README.md")))
        @test !occursin("root workspace", _readtext(joinpath(root, "README.md")))
    end

    @testset "Aqua" begin
        # Ambiguities are checked across the family below (Aqua's own check
        # would also report Base/stdlib ones)
        Aqua.test_all(ERGMMulti; ambiguities=false)
        @test isempty(Test.detect_ambiguities(ERGMMulti))
        @test isempty(Test.detect_ambiguities(ERGMMulti, ERGM, NetworkCore))
    end

    @testset "Workflows reconstruct the ecosystem layout from [sources]" begin
        pkgdir = dirname(@__DIR__)
        PKG = "ERGMMulti"
        EXPECTED = Set(["ERGMMulti.jl", "ERGM.jl", "NetworkCore.jl"])
        siblings = sort!(collect(setdiff(EXPECTED, ["$PKG.jl"])))
        in_layout = !isempty(sibling_checkouts(pkgdir, siblings))
        in_layout || @info "The workflows' layout step is not run: none of the " *
                           "sibling checkouts $(join(siblings, ", ")) is beside " *
                           "$(dirname(pkgdir)) (a lone checkout or a registry " *
                           "install, where [sources] is not used)."
        # The predicate itself: no sibling → skip; one sibling → run (and a
        # missing other sibling then fails the set comparison below)
        mktempdir() do root
            lone = joinpath(root, "$PKG.jl")
            mkpath(lone)
            @test isempty(sibling_checkouts(lone, siblings))
            mkpath(joinpath(root, first(siblings)))
            touch(joinpath(root, first(siblings), "Project.toml"))
            @test sibling_checkouts(lone, siblings) == [first(siblings)]
        end
        for wf in ("CI.yml", "Documentation.yml")
            yml = _readtext(joinpath(pkgdir, ".github", "workflows", wf))
            @test !occursin(r"for pkg in", yml)              # no hand-kept clone list
            @test !occursin("checkout_sources.jl", yml)
            @test occursin("path: $PKG.jl\n", yml)
            step = match(r"\n      - name: Reconstruct the ecosystem layout from \[sources\]\n        shell: julia[^\n]*\n        run: \|\n((?:          [^\n]*\n)+)", yml)
            @test step !== nothing
            step === nothing && continue
            @test first(findfirst("setup-julia", yml)) < step.offset
            if !in_layout
                @test_skip "layout step of $wf (no sibling checkout)"
                continue
            end
            # Run the workflow's own step without cloning: in the layout this
            # suite runs in, it must find exactly the siblings [sources] names.
            script = replace(step.captures[1], r"^          "m => "")
            out = mktemp() do path, io
                write(io, script); close(io)
                withenv("GITHUB_WORKSPACE" => dirname(pkgdir),
                        "GITHUB_REPOSITORY" => "statistical-network-analysis-with-Julia/$PKG.jl",
                        "LAYOUT_CHECK_ONLY" => "true", "GITHUB_STEP_SUMMARY" => nothing,
                        # `Pkg.test` runs this suite with a sandbox load path
                        # that hides the stdlibs the step loads (`TOML`); the
                        # workflow runs it with Julia's default load path
                        "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing) do
                    run_fresh(`$(Base.julia_cmd()) --startup-file=no $path`).out
                end
            end
            @test Set(m.captures[1] for m in eachmatch(r"^\| (\S+\.jl) \|"m, out)) == EXPECTED
        end
        # One matrix cell runs the tests on four threads so the bootstrap's
        # thread-count-independence pin exercises the threaded path
        ci = _readtext(joinpath(pkgdir, ".github", "workflows", "CI.yml"))
        @test occursin("JULIA_NUM_THREADS: \${{ (matrix.version == '1' && matrix.os == 'ubuntu-latest') && '4' || '1' }}", ci)
        # The package and docs environments source the same siblings by path
        project = _readtext(joinpath(pkgdir, "Project.toml"))
        block = match(r"\[sources\]\n((?:[^\[]*\n)*)", project)
        @test block !== nothing
        sources = [String(mt.captures[1]) for mt in
                   eachmatch(r"^(\w+)\s*=\s*\{\s*path\s*="m, block.captures[1])]
        @test sort(sources) == ["ERGM", "NetworkCore"]
        docs = _readtext(joinpath(pkgdir, "docs", "Project.toml"))
        for pkg in sources
            @test occursin("$pkg = {path = \"../../$pkg.jl\"}", docs)
        end
        @test occursin("ERGMMulti = {path = \"..\"}", docs)
        # The docs environment pins its dependencies
        @test occursin(r"\[compat\][\s\S]*Documenter = \"1\"", docs)
    end

    @testset "Every export has a docstring with a runnable example" begin
        # Every export has a docstring with a runnable example. The Documenter build (checkdocs=:exports) checks presence,
        # not content, so walk the docsystem: every ERGMMulti-owned docstring
        # of an exported binding — including the ones ERGMMulti attaches to
        # the shared NetworkCore/StatsAPI generics (`gof`, `coef`, `confint`, ...)
        # — must contain a fenced ```julia block, and every such block must
        # RUN in a fresh module that has done nothing but `using ERGMMulti`
        # (so an example that needs `ERGM`, `NetworkCore`, `Random` or
        # `LinearAlgebra` says so itself). Warnings the examples deliberately
        # provoke (a boundary statistic, an excluded bootstrap replicate) go to
        # a null logger.
        # Mirrors ERGM.jl's and NetworkCore.jl's testsets of the same name.
        meta_multi = Base.Docs.meta(ERGMMulti)
        documented_elsewhere(b) = any(haskey(Base.Docs.meta(m), b)
                                      for m in (NetworkCore, ERGM, StatsAPI))
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
        @test length(blocks) >= 38
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

    @testset "Precompile workload is declared" begin
        root = joinpath(@__DIR__, "..")
        project = _readtext(joinpath(root, "Project.toml"))
        @test occursin("PrecompileTools = \"aea7be01-6a6a-4083-8856-8a6e6704d82a\"", project)
        @test occursin(r"\[compat\][\s\S]*PrecompileTools = \"1\"", project)
        src = _readtext(joinpath(root, "src", "ERGMMulti.jl"))
        @test occursin("@setup_workload begin", src) && occursin("@compile_workload begin", src)
        # The workload exercises both directednesses and every documented
        # first-session call
        for needle in ("for _pc_directed in (true, false)", "fit_ergm_multi(_pc_m, _pc_terms;",
                       "simulate_multi_ergm(_pc_m", "gof(_pc_fit;", "se=:bootstrap")
            @test occursin(needle, src)
        end
        # ... and the model it compiles is the documented first model
        @test occursin("WithinLayer(NodeMatch(:grp), 1)", src)
    end
end
