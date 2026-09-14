# Changelog

All notable changes to ERGMMulti.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

Release driven by the 2026-07 expert-panel review: the multilayer data model
is rebuilt around R `ergm.multi`'s block-diagonal construction with
integer-indexed layers, the placeholder estimator is replaced by a real
within-layer-dyad MPLE, and sampling no longer drops network attributes.

### Breaking

- **`MultilayerNetwork{D}`, `MultiERGMModel{D}` and `MultiERGMResult{D}` carry
  the directedness as a type parameter** (panel 2026-09, item 19), as
  `Network{T,D}` and `ERGM.ERGMModel{T,D}` do. `MultilayerNetwork(n;
  directed=true)` returns a `MultilayerNetwork{true}` whose `layers` field is
  a concretely typed `Vector{Network{Int,true}}`; the `directed::Bool` field
  is gone. *Migration:* read `is_directed(m)` instead of `m.directed`
  (`m.directed` now throws). `add_layer!` refuses a layer of the other
  directedness with an `ArgumentError` naming both. `MultiERGMResult` gains a
  `boot_replicates` field (positional construction passes every field — the
  struct has no other constructor).
- **`MultiERGMModel.terms` is an `ERGM.TermSet`** (tuple-backed, with the
  coefficient labels in `.names`), not a `Vector{AbstractERGMTerm}`; its
  constructor is `MultiERGMModel(terms, m; offsets=...)` and it **validates
  the formula** (below). `ergm_multi` and `simulate_multi_ergm` both go
  through it and also accept a prebuilt model.
- **Model construction validates the formula against the data** (item 12
  inheritance; mirrors `ERGMModel`'s inner constructor). What used to be a
  silently wrong fit is now an `ArgumentError` before any fitting:
  - a layer index the network does not have (`LayerEdges(3)`,
    `InterlayerDependence(1, 3)`, `WithinLayer(t, 3)` on two layers — which
    used to yield all-zero coefficients, `NaN` standard errors and
    `converged=false`) throws "term 'L.edges.3' refers to layer 3, but the
    network has 2 layers (:friendship, :advice); layer indices must lie in
    1:2";
  - every `WithinLayer(t, l)` is validated against layer `l` with the
    now-public `ERGM._validate_formula`: a missing vertex attribute
    (`WithinLayer(NodeMatch(:welth), 1)` used to compute `0.0`) throws "term
    'nodematch.welth' refers to vertex attribute :welth, which does not exist
    on the network …", a partial attribute "… needs vertex attribute :grp on
    every vertex, but 1 of 4 vertices have no value … statnet refuses NA
    attribute values", an `EdgeCov` of the wrong size names both sizes, and
    an undirected-only term on a directed layer (`Kstar`, `GWDegree`,
    `Degree` — undirected-only in ERGM 0.2 as in R) throws "term 'kstar2' is
    only defined for undirected networks, but the network is directed. Use
    `OStar(2)` / `IStar(2)` instead …"; the reverse for `Mutual`, `OStar`,
    `IStar`, `GWODegree`, `GWIDegree` on an undirected layer;
  - `LayerMutual` and `MultiplexMutual` declare `requires_directed` and are
    refused on a `MultilayerNetwork{false}` — at construction and in
    `compute`/`change_stat_layer` — with ERGM's sentence "term 'L.mutual.all'
    is only defined for directed networks, but the network is undirected …"
    (both used to return `0.0` silently);
  - offsets must index terms, must be finite, and must leave at least one
    coefficient free (moved from `ergm_multi` into the constructor);
  - (round 2) a **bare ERGM.jl term** — `Mutual()` where
    `WithinLayer(Mutual(), l)` was meant, ergm.multi's `L(~mutual, ~A)` — or
    a multilevel descriptive (`Nestedness(1)`) is refused by the constructor
    with "term 'mutual' (Mutual) has no multilayer change statistic
    (`change_stat_layer`): … lift an ERGM.jl term into a layer with
    `WithinLayer(Mutual(...), l)` …" (the generic fallback checks
    `hasmethod(change_stat_layer, …)`); it used to construct and die inside
    the design builder with `MethodError: no method matching
    change_stat_layer(::Mutual, …)`;
  - (round 2) a layer that **contains a self-loop** is refused
    (`_refuse_layer_self_loops`, the counterpart of `ERGM.ERGMModel`'s
    `_refuse_self_loops`): "layer :b (layer 2) contains 1 self-loop (at
    vertex 1). ERGMMulti models the off-diagonal dyads of every layer only …
    (R ergm warns \"This network contains loops\" here). Remove the loops
    (`rem_edge!(layer_network(m, 2), v, v)`) or build the layer with
    `loops=false`". It used to be accepted: `compute(LayerEdges(l), m)`
    counted the loop while the MPLE design skipped it (`logit(2/18)` for 3
    ties in 20 dyads) and `gof` compared the observed count against
    simulations that can never hold a loop. A `loops=true` layer with no
    loop is accepted, as in ERGM.jl.
- **A two-mode (bipartite) layer is refused by `add_layer!`** (hence by
  `as_multilayer` and every `MultilayerNetwork`): "layer :a is two-mode
  (bipartite); ERGMMulti models one-mode layers only — the within-layer dyad
  universe would enumerate the impossible within-mode pairs as observed
  non-ties, so the fit is refused instead (statnet's bipartite layer terms
  b1dspL, b2dspL, ... are not implemented …)". Bipartite layers used to enter
  the one-mode dyad universe silently: `LayerEdges` on two `network(6;
  bipartite=3)` layers with 2 ties each fitted `logit(2/15)` over 15 dyads
  per layer instead of the bipartite `logit(2/9)`, and the "Not implemented"
  paragraphs claimed there was nothing to misuse (round 2).
- **A layer selected by name inside a term is an `ArgumentError`, not a
  `MethodError`** (round 2). `LayerEdges(:advice)`, `LayerMutual([:a, :b])`,
  `LayerTriangle(:a)`, `WithinLayer(Triangle(), :advice)`,
  `InterlayerDependence(:friendship, :advice)` and `MultiplexMutual(:a, 2)`
  throw "LayerEdges selects layers by index, not by name (got :advice): use
  LayerEdges(k) with k = findfirst(==(:advice), layer_names(m)) — the index
  of that layer in `layer_names(m)` …" — the migration note below, at the
  point where a user meets it — instead of a `MethodError` with a "Closest
  candidates" dump.
- **`gof` and `simulate_multi_ergm` refuse a non-finite coefficient**
  (round 2). After a boundary drop (`coef(fit)[k] == -Inf`), `gof(fit)`,
  `simulate_multi_ergm(fit.model, coef(fit))` and `simulate_multi_ergm(m,
  terms, θ)` with a `∓Inf`/`NaN` entry throw "gof: every coefficient must be
  finite (got duplex.1.2 = -Inf): a multilayer network cannot be simulated at
  an infinite or NaN coefficient — the chain would reject every proposal and
  return copies of the observed network … remove the dropped term as R's
  drop=TRUE does and refit". They used to run that frozen chain silently
  (`θ'Δ = -Inf·0 = NaN`, `log(u) < NaN` always false): every "simulated"
  network was the observed one and `gof` reported `p = 1.0` for every
  statistic with no warning. `se=:bootstrap` already refused the case.
- **Within-layer attribute terms are materialized and expanding terms
  expand.** `WithinLayer(NodeMatch(:g), l)` stores ERGM.jl's materialized
  twin (a dense attribute snapshot read per toggle, not a `Dict`); a
  multi-level `NodeFactor`, a multi-cell `NodeMix` or a `Degree(0:2)` inside
  `WithinLayer` becomes one statistic per level/cell/degree, exactly as in
  ERGM.jl, so `length(model.terms)` can exceed the number of terms given.
  `offsets` keys are remapped to the expanded positions; an offset on a term
  that expands into several statistics is refused with the fix
  (`NodeFactor(:attr; level=x)`, `Degree(d)`). `simulate_multi_ergm`'s `θ`
  has one entry per statistic. (Previously `WithinLayer(NodeFactor(:g), l)`
  computed the *sum* of the levels under one coefficient.)
- **Coefficient labels are direction-aware** (`name(term, m)`): a directed
  layer's `WithinLayer(GWESP(0.5), 1)` / `WithinLayer(GWDSP(0.5), 1)` is
  labelled `L1.gwesp.OTP.fixed.0.5` / `L1.gwdsp.OTP.fixed.0.5` (R's label), an
  undirected one `L1.gwesp.fixed.0.5`, matching ERGM.jl and R. `show`, `gof`
  panels and `coeftable` all read `model.terms.names`.
- **A statistic at the boundary of its attainable range gets R's `drop`
  semantics** instead of a Newton iteration that "converged" on the flat
  asymptote (−21 with a standard error of 19,000 on the test case). Read
  off the design with ERGM.jl's `_boundary_columns_iterated`: the coefficient
  is fixed at `-Inf`/`+Inf` with standard error 0 and p-value 0, R's warning
  ("observed statistic(s) duplex.1.2 are at their smallest attainable values.
  Their coefficients will be fixed at -Inf …") is emitted, the remaining
  coefficients are estimated on the dyads the dropped statistic does not touch
  (the exact limit of the pseudo-likelihood; `bic` uses that dyad count),
  `dof` counts only finite free coefficients, `is_exact(fit)` is `false`, and
  `show`/`approximations` carry a note. A bootstrap refit that lands on a
  boundary is a non-finite replicate and is excluded (below).
- **`se=:bootstrap` is refused on a fit with a `∓Inf` coefficient**
  (`ArgumentError`: "ergm_multi: se=:bootstrap is not available when a
  coefficient is fixed at ±Inf by a statistic at the boundary of its
  attainable range (duplex.1.2): a multilayer network cannot be simulated at
  an infinite coefficient. Remove the term (as R's drop=TRUE does) or keep
  the default se=:hessian …"), as `ERGM.mple` does.
- **Perfect separation by a combination of statistics is detected and the
  fit returned unconverged**, with or without offsets. No single column at
  its boundary, yet a linear combination of them perfectly predicts the
  within-layer ties — R's "The MPLE does not exist!". ERGM.jl's asymptote
  test (`ERGM._separated`) runs after Newton stops; when it fires, R's
  sentence is printed ("ergm_multi: the MPLE does not exist (perfect
  separation): the pseudo-likelihood has no finite maximum, and the returned
  coefficients are the point at which Newton stopped on its flat asymptote
  …"), `fit.converged` is `false`, the new `fit.separated` field is `true`,
  `is_exact(fit)` is `false`, and `show`/`approximations` print the
  separation verdict instead of the generic non-convergence caveat. Such a
  design used to be returned with `converged == true` and coefficients in
  the tens (standard errors in the tens of thousands).
- **`is_exact(fit)` is `false` for an unconverged fit and for one with a
  `∓Inf` coefficient**, as in ERGM.jl.
- **`simulate_multi_ergm` refuses masked networks.** The chain toggles every
  within-layer dyad of every layer, so a masked (unobserved) dyad used to be
  toggled at its face value. Every layer now goes through `require_observed`
  (`ArgumentError` "simulate_multi_ergm (layer 1) does not support missing
  (unobserved) dyads, but the network has 1 masked dyad …"), and
  `Networks.missing_policies(simulate_multi_ergm) == (:error,)` — no
  `missing=` keyword, as for `ergm_multi`. `gof` is covered transitively (a
  `MultiERGMResult` never holds a masked network).
- **`MultiERGMResult` gains a `separated::Bool` field** (after
  `boot_replicates`); positional construction must pass it. The struct's
  only constructor takes all eleven fields: the round-2 8-/9-/10-argument
  "compatibility" constructors, which defaulted `se_type`, `boot_replicates`
  and `separated`, are removed (round 3) — no released layout ever had those
  shapes (0.1.0's `MultiERGMResult` was a different struct), and they let a
  result be built whose flags did not describe the numbers it carried.
- **`add_layer_edge!` refuses an actor id outside `1:n` and a self-loop on
  a layer built without `loops=true`** (round 3), with an `ArgumentError`
  naming the ids and the range ("add_layer_edge!: actor ids must lie in
  1:10 (got (0, 2)); the multilayer network has 10 actors. Ids are 1-based —
  a 0 usually means 0-based data — and an edge is never dropped silently.";
  "add_layer_edge!: (3, 3) is a self-loop, and layer :a was built without
  `loops=true` …"). It used to discard `add_edge!`'s `false` and return the
  network unchanged, so a mistyped or 0-based id gave a silently smaller
  network and a fit on it. A layer that allows loops still accepts the loop
  here; `MultiERGMModel` is where a loop is refused.
- **`layer_network(m, l::Int)` with an out-of-range index is an
  `ArgumentError`** naming the layers ("layer index 5 out of range: the
  network has 2 layers (:friendship, :advice); layer indices must lie in
  1:2 (see `layer_names(m)`)."), not a raw `BoundsError` (round 3); the
  `Symbol` form and the term validator already named them. `add_layer_edge!`
  with a bad layer index inherits the sentence.
- **A network with no within-layer dyads is refused by `MultiERGMModel`**
  (round 3): `MultilayerNetwork(0)` or `MultilayerNetwork(1)` with a layer
  used to run Newton on an empty design and return `coef = [0.0]`, `NaN`
  standard errors, `nobs = 0`, `converged = false` and two misleading
  warnings (Networks' "Hessian at the solution is not negative definite" and
  the "raise `maxiter`" non-convergence caveat). It is now an `ArgumentError`
  at construction — "the network has no within-layer dyads (n = 1 actor,
  1 layer); a multilayer ERGM needs at least two actors — nothing to
  estimate or simulate" — from `ergm_multi` and `simulate_multi_ergm` alike.
- **The boundary-statistic warning no longer says "R ergm reports the
  same"** (round 3). `ergm_multi` used to print `ERGM._warn_boundary`'s
  sentence, whose closing parenthesis is true of `ergm` and false of
  `ergm.multi` 0.3.0 (which does *not* drop: it warns "The MPLE does not
  exist!" and returns a finite value — fixture section (iii)). ERGMMulti's
  `_warn_multi_boundary` emits the same R sentence through ERGM.jl's now
  `public` `_warn_boundary(...; note=)`, closing with "(no finite maximum
  pseudo-likelihood estimate exists; R ergm's drop=TRUE; ergm.multi 0.3.0
  warns \"The MPLE does not exist!\" and returns a finite value with a
  standard error in the thousands instead — see the estimation guide)" —
  the parenthesis is the one variable clause, never a restated sentence
  (reconciliation round: `_boundary_columns_iterated`, `_separated`,
  `_warn_separated` and `_warn_boundary` are `public` in ERGM.jl and the
  reach-in testset asserts it instead of holding them `@test_broken`). A
  test that matched the old parenthesis must match the new one.
- **`fit_multi_ergm` is deprecated** in favour of the harmonised
  `fit_ergm_multi` (item 16): it is now a wrapper that emits a deprecation
  warning and forwards to `ergm_multi`; `fit_multi_ergm === ergm_multi` no
  longer holds.
- **Sampler defaults are dyad-scaled.** `simulate_multi_ergm`, `gof` and the
  bootstrap's `boot_burnin`/`boot_interval` default to `nothing`, resolved by
  the ERGM family's one rule (`ERGM._mcmc_defaults` over the within-layer
  dyads: `burnin = 20 × n_dyads`, `interval = max(100, n_dyads ÷ 10)`) instead
  of the fixed `1000`/`100`. Seeded draws differ from 0.1 (the kernel is now
  `ERGM.mh_toggle!`; the proposal-then-acceptance rng order is unchanged).
- **`MultilayerNetwork` restructured and de-parametrized.** Previously
  `MultilayerNetwork{T}` held `layers::Dict{Symbol,Network}` and the
  constructor pre-populated layers by name; now `MultilayerNetwork(n;
  directed=true)` starts empty with `layers::Vector{Network{Int}}` plus
  `layer_names`. *Migration:* build with `MultilayerNetwork(n)` then
  `add_layer!(m, :name)`; fetch layers via `layer_network(m, idx_or_name)`
  instead of dict indexing.
- **Layers are selected by integer index, not `Symbol`.** Term constructors
  take `LayerSel = Union{Int,Vector{Int},Colon}`: `LayerEdges(2)` or
  `LayerEdges()` (pools all layers), not `LayerEdges(:friendship)`.
  *Migration:* replace layer symbols with layer indices (see
  `layer_names(m)` for the mapping).
- **Term protocol renamed to `change_stat_layer(term, m, l::Int, i, j)`**
  (was `change_stat(term, mln, layer::Symbol, i, j)`), using the
  ecosystem-wide state-independent add-direction convention. *Migration:*
  custom multilayer terms must implement `change_stat_layer` and drop
  toggle-sign logic.
- **`combine_networks` repurposed.** It no longer set-combines a
  `Vector{Network}` by `:union`/`:intersection`; it now flattens a
  `MultilayerNetwork` into the block-diagonal n×L `Network` (with `:layer`
  and `:actor` vertex attributes) that estimation runs on. *Migration:* the
  union/intersection helper is gone — combine graphs manually if needed.
- **`as_multilayer` / `split_by_layer` signatures changed.**
  `as_multilayer(nets::Vector{<:Network}, names::Vector{Symbol})` replaces
  the `Dict{Symbol,Network}` form; `split_by_layer(combined, n, L; names)`
  now rebuilds a `MultilayerNetwork` from a block-diagonal network.
  *Migration:* pass a vector plus names; adjust `split_by_layer` call sites.
- **`simulate_multi_ergm` signature and return changed** to
  `simulate_multi_ergm(m, terms, θ; n_sim, burnin, interval, rng) ->
  Vector{MultilayerNetwork}` (was `(result; n_sim, burnin)` returning
  unsampled deep copies). *Migration:* pass network, terms, and
  coefficients explicitly.
- **`ergm_multi` accepts a `MultilayerNetwork` only** (no
  `MultiNetwork`/`MultilevelNetwork` methods, no `method=:mple` keyword —
  MPLE is the estimator). *Migration:* convert data via `as_multilayer`;
  drop `method=`.
- **Removed exports:** `LayerSpec`, `LevelSpec`, `LayerLogic`,
  `BetweenLayers` (no replacements); `CrossLevelEdge` lost its
  `(level1, level2)` fields; `WithinLayer` now takes an integer layer.
  *Migration:* re-express models with `WithinLayer(term, layer::Int)` and
  the current term set.
- **Statistic labels renamed** to `ergm.multi`-style names (`L.edges.<sel>`,
  `duplex.<l1>.<l2>`, `duplex.mutual`, `crossnet.edges`, `nestedness`,
  `crosslevel.edges`). *Migration:* update code that matches coefficient
  names.
- **Minimum Julia raised to 1.12**; package UUID regenerated. *Migration:*
  upgrade Julia and re-resolve environments pinning the old UUID.

### Added

- **Fixture section (iv): the undirected layers, pinned against ergm.multi**
  (`test/fixtures/r/twolayer_layer_terms.R`, round 3). Sections (i)–(iii)
  are directed, and the undirected branch — unordered within-layer dyads
  (`i < j`, `nobs = L·n(n−1)/2`), `InterlayerDependence` counting unordered
  co-occurrences, the undirected `triangle`/`gwesp`/`kstar`/`gwdegree`
  forms — was validated only by brute force and the analytic
  `logit(density)` test. The script now also draws two **undirected**
  12-actor layers from the same seed stream (frozen as edge lists), freezes
  `summary(lnu ~ L(~edges,~A) + L(~edges,~B) + L(~edges,~A&B) +
  L(~triangle,~A) + L(~gwesp(0.5,fixed=TRUE),~A) + L(~nodematch("grp"),~B)
  + L(~kstar(2),~B) + L(~gwdegree(0.3,fixed=TRUE),~B))` (26, 18, 7, 10,
  24.670…, 8, 49, 15.141…; tolerance 1e-10) and the dyad-dependent MPLE
  `ergm(lnu ~ L(~edges,~A) + L(~edges,~B) + L(~edges,~A&B) +
  L(~triangle,~A), estimate = "MPLE")` as shipped plus its exact `glm`
  refit at epsilon 1e-14 on `ergmMPLE(output = "matrix")`'s design, whose
  weights sum to 132 = `nobs`. ERGMMulti.jl agrees with the exact refit to
  ~2e-11 in the coefficients and ~6e-10 in the standard errors (against
  1e-6), and labels the undirected gwesp `L1.gwesp.fixed.0.5` as R does. The
  Terms guide's correspondence table now says which fixture section
  (directed / undirected) pins each row.
- **`Base.show` for `MultiERGMModel`, `MultiNetwork` and `MultilevelNetwork`**
  (round 2): one-line summaries consistent with `ERGM.ERGMModel`'s —
  `MultiERGMModel{true}: 4 actors, 2 directed layers (:friendship, :advice);
  terms: L.edges.1 + L.edges.2; offsets: 2 => -2.0`, `MultiNetwork: 2
  networks — x (3 vertices, 1 edges, directed); y (…)`, `MultilevelNetwork:
  2 levels (level 1: 5 nodes, 3 edges; level 2: …), 1 cross-level edge` —
  instead of the default struct dump (the `TermSet` tuple, `Network{Int64}[…]`
  vectors, membership `Dict`s). Pinned with `sprint(show, x)` tests.
- **Fixture section (iii): ergm.multi's behaviour on a boundary statistic,
  frozen as a documented divergence** (`test/fixtures/r/twolayer_layer_terms.R`,
  round 2). On the 6-actor cycle/reversed-cycle design where `L(A&B)~edges`
  is 0, R ergm.multi 0.3.0 (R 4.6.1, ergm 4.12.0) warns "The MPLE does not
  exist!" and returns a *finite* −17.58 with standard error 1996 for the
  boundary term (its layer operator does not propagate the attainable range
  to `ergm.checkextreme.model`, so ergm's `drop=TRUE` never fires); its
  finite coefficients are `logit(6/18)` exactly. ERGMMulti.jl fixes the
  boundary term at `-Inf` and is held to R's finite coefficients at 1e-6.
  The docs, the `show` note and the source comment that said `ergm.multi`
  "inherits" or "drops the term the same way" now state this.
- **Every export carries a docstring with a runnable example, and the test
  suite runs them** (grade-A criterion 5). Examples were added to
  `MultiNetwork`, `MultilevelNetwork`, `add_layer_edge!`, `layer_network`,
  `add_cross_level_edge!`, every term (`LayerEdges`, `LayerMutual`,
  `LayerTriangle`, `InterlayerDependence`, `MultiplexMutual`,
  `CrossNetEdges`, `Nestedness`, `CrossLevelEdge`, `LevelHomophily`),
  `change_stat_layer`, `MultiERGMResult`, `fit_multi_ergm` and `gof`; the
  previously undocumented **`layer_names`** is documented; and the StatsAPI
  accessors `coef`, `stderror`, `vcov`, `loglikelihood`, `nobs`, `dof`,
  `aic`, `bic` gain `MultiERGMResult` docstrings whose examples are
  closed-form checks on a 4-actor network (`coef ≈ logit(density)` per
  layer, the binomial information for `stderror`, the two-binomial
  `loglikelihood`, `nobs == 24`). The new testset "Every export has a
  docstring with a runnable example" walks `Base.Docs.meta(ERGMMulti)` for
  `names(ERGMMulti)`, requires a fenced ```julia block on every
  ERGMMulti-owned docstring (39 blocks) and executes each in a fresh module
  that has done nothing but `using ERGMMulti`, so an example that rots
  fails `Pkg.test()`.
- **The multilayer adapters honour the conversion contract** (panel 2026-09,
  item 4; Networks.jl `docs/src/guide/conversion_invariants.md`). The
  block-diagonal `Network` CAN represent a missing-dyad mask, so
  `combine_networks(m)` now **preserves it**: a masked (unobserved) dyad
  `(i, j)` of layer `l` is masked at `((l-1)·n + i, (l-1)·n + j)` of the
  combined network (`n_missing_dyads(combined)` is the sum over the layers),
  and `split_by_layer` carries every mask back onto its layer. Until now the
  combined network was built from the edge sets alone and reported zero
  masked dyads — an unobserved tie silently became an observed absent one
  (the "I — read at face value" cell of the invariant table). A masked
  cross-block dyad is refused by `split_by_layer` with the same
  `ArgumentError` as a cross-block edge ("combined network has a masked
  cross-block dyad (1, 6): cross-block dyads are structurally empty in the
  block-diagonal construction and cannot be unobserved"); the `loops` flag
  travels both ways; `split_by_layer` checks it was given one name per layer.
  All three adapters take **`report=true`** and then return `(result,
  ::ConversionReport)`: `combine_networks` names every vertex, edge and
  network attribute of every layer that the block-diagonal network cannot
  hold (it carries only `:layer`/`:actor`; each entry's reason names the
  layer); `split_by_layer` names the
  combined network's own vertex attributes other than `:layer`/`:actor`,
  its edge and network attributes; `as_multilayer` stores the layers as
  given and its report is lossless. `supports_missing` is `true` for all
  three — none reads a face value — and none offers a `missing=` keyword
  (`missing_policies == (:error,)`), since there is nothing to opt into: the
  guard stays on the estimator, which still refuses a masked layer.
- **PrecompileTools workload** (item 18): the documented first session —
  build a two-layer network, `fit_ergm_multi` with `[LayerEdges(1),
  LayerEdges(2), WithinLayer(NodeMatch(:grp), 1), InterlayerDependence(1, 2)]`,
  `coeftable`/`confint`/`show`, `simulate_multi_ergm`, `gof`, a 3-replicate
  `se=:bootstrap` — on a directed AND an undirected `MultilayerNetwork{D}`
  (two specializations of every kernel), under a devnull console logger.
  Measured numbers are in the Performance section.
- **CI runs one matrix cell on four threads** (`JULIA_NUM_THREADS: 4` on
  Julia 1 / ubuntu, items 7/29), so the bootstrap's thread-count-independence
  pin (`threaded=true` vs `threaded=false` under one `rng` give bit-identical
  standard errors) exercises the threaded path. A testset asserts that the
  `for pkg in …` clone lists of `CI.yml` and `Documentation.yml` equal
  exactly the `[sources]` keys of `Project.toml` (`Networks ERGM`, foundation
  first) and that `docs/Project.toml` sources the same siblings, so the
  workflows cannot drift from the actual dependency layout.
- Tests: the sampler is pinned **literally** against the pre-kernel loop —
  edge lists of three seeded draws on the directed fixture (`LayerEdges(1),
  LayerEdges(2), LayerMutual(), InterlayerDependence(1, 2)`, `Xoshiro(7)`)
  and three on the undirected one, computed by a verbatim replica of the
  committed loop, so a change to the proposal, the acceptance rule or the
  rng order in either package fails as a literal mismatch; an 8-term directed
  model (`LayerEdges(1), LayerEdges(2), LayerMutual(), InterlayerDependence(1, 2),
  MultiplexMutual(1, 2), WithinLayer(GWESP(0.5), 1), WithinLayer(OStar(2), 2),
  WithinLayer(TwoPath(), 1)`) whose generated change-statistic fill costs
  exactly 0 bytes and whose sampler adds < 4 bytes per step (Graphs.jl's
  adjacency growth, ERGM.jl's own bound), plus a 34-statistic model (past
  Base's 32-element `map` fallback) at 0 bytes; the conversion-contract
  testset above; the clone-list and workload testsets.
- **`fit_ergm_multi`** — the ecosystem's canonical `fit_<model>` name, a
  `const` alias of `ergm_multi` (`fit_ergm_multi === ergm_multi`), exported
  alongside it (item 16). `Networks.missing_policies(ergm_multi) == (:error,)`
  is declared explicitly: the estimator offers no `missing=` keyword.
- **Full StatsAPI surface on `MultiERGMResult`** (item 15): `confint(fit;
  level=0.95)` (normal-theory limits; `NaN` rows for offsets) and
  `coeftable(fit)` (a `Networks.CoefficientTable` — exactly the table `show`
  prints, offset rows tagged ` (offset)`), joining `coef`, `stderror`, `vcov`,
  `loglikelihood`, `nobs`, `dof`, `aic`, `bic`; pinned by
  `Networks.check_statsapi(fit; strict=true)` on a Hessian, a bootstrap and an
  offset fit.
- **`MultiERGMModel(terms, m; offsets)`** as a public, validating constructor;
  `ergm_multi(model; kwargs...)` and `simulate_multi_ergm(model, θ; kwargs...)`
  accept it, so a validated model can be reused. `is_directed` is defined on
  `MultilayerNetwork`, `MultiERGMModel` and `MultiERGMResult`.
- **`name(term::WithinLayer, m::MultilayerNetwork)`** — the direction-aware
  coefficient label (a method of ERGM.jl's two-argument `name(term, net)`
  generic).
- **`ERGM.has_dyad_dependent(::MultiERGMModel)`** — the multilayer method of
  ERGM.jl's shared predicate, behind `show`'s caveat, `is_exact` and
  `approximations` (replaces the private `_has_dyad_dependent`).
- **Loud non-convergence.** A Newton iteration that does not converge emits a
  warning, `show` prints the caveat right under `converged: false`, and
  `approximations(fit)` lists it.
- **Bootstrap replicates without a finite MPLE are excluded, once, out loud.**
  A `se=:bootstrap` refit that does not converge or lands on a boundary
  statistic is a `NaN` row of the new `fit.boot_replicates` matrix, excluded
  from the covariance; one warning reports the count and `show`/
  `approximations` carry a note (fewer than 2 finite refits is an error).
  Previously such a refit "converged" on the flat asymptote and inflated
  every standard error (the `LayerMutual` bootstrap SE on the test fixture
  fell from 2.85 to 0.67).
- **`ergm_multi(...; threaded=true)`** passes through to
  `Networks.bootstrap_cov`: the bootstrap refits run on all threads by
  default, on one with `threaded=false`. The replicates are simulated from
  `rng` before any refit and each refit is deterministic, so the standard
  errors are identical whatever the thread count — pinned by a test that
  compares `threaded=true` and `threaded=false` under the same `rng`.
- **Second provenanced golden fixture: `ergm.multi`'s layer terms and a
  dyad-dependent MPLE** (`test/fixtures/twolayer_layer_terms.toml`,
  regenerable with `Rscript test/fixtures/r/twolayer_layer_terms.R >
  test/fixtures/twolayer_layer_terms.toml`; ergm.multi 0.3.0 / ergm 4.12.0 /
  R 4.6.1). It reuses the SAME frozen layers as `twolayer_ergm_multi.toml`
  (identical generating code and seed 20260713; the testset asserts the edge
  lists agree), wrapped as `lnw <- Layer(list(A = net1, B = net2))`.
  (i) `summary(lnw ~ L(~edges,~A) + L(~edges,~B) + L(~mutual,~A) +
  L(~triangle,~A) + L(~edges,~A&B) + mutualL(Ls=list(~A,~B)) +
  L(~gwesp(0.5,fixed=TRUE),~A) + L(~ostar(2),~B) + L(~nodematch("grp"),~B) +
  L(~twopath,~A))` is frozen exactly (tolerance 1e-10) and asserted against
  `LayerEdges(1)`, `LayerEdges(2)`, `LayerMutual(1)`, `LayerTriangle(1)`,
  `InterlayerDependence(1, 2)`, `MultiplexMutual(1, 2)`,
  `WithinLayer(GWESP(0.5), 1)`, `WithinLayer(OStar(2), 2)`,
  `WithinLayer(NodeMatch(:grp), 2)` and `WithinLayer(TwoPath(), 1)` — all
  ten agree exactly. Both cross-layer terms count ORDERED dyads in R
  (verified on a 4-actor example where ordered and unordered counting
  differ), as `InterlayerDependence` and `MultiplexMutual` already did, so
  no term definition changed. (ii) The DYAD-DEPENDENT MPLE `ergm(lnw ~
  L(~edges,~A) + L(~edges,~B) + L(~edges,~A&B) + L(~mutual,~A),
  estimate="MPLE")` — coefficients and standard errors as shipped, plus the
  same pseudo-likelihood refit at `glm` epsilon 1e-14 on
  `ergmMPLE(output="matrix")`'s own compressed design (response / predictor
  / weights; the weights are asserted to sum to the 760 within-layer dyads).
  `ergm_multi(m, [LayerEdges(1), LayerEdges(2), InterlayerDependence(1, 2),
  LayerMutual(1)])` matches the exact refit to ~9e-8 in the coefficients and
  ~7e-9 in the standard errors (tolerance 1e-6); the as-shipped tolerances
  (1e-6 / 1e-4) are 20x the measured glm slack, not hand-chosen. This is the
  fixture that licenses the cross-layer and mutual CHANGE STATISTICS
  against R's: the MPLE is a deterministic function of the change-statistic
  design.
- Tests: the README's pre-0.2 single-edge design (30 actors, one friendship
  tie, empty advice layer) pinned to R's drop semantics (`coef[2] == -Inf`,
  `coef[1] ≈ log(1/1738)`, `dof == 1`, `bic` on 1739 dyads, bootstrap
  refused); a separated `WithinLayer(NodeMatch(:grp), 1)`; ERGM.jl's
  separated-by-a-combination design lifted into a layer; a rank-deficient
  `[LayerEdges(1), LayerEdges(1)]` design (NaN standard errors, Networks'
  "not negative definite" warning, `converged == false`); a `LayerTriangle`
  bootstrap with excluded replicates (one warning, `NaN` rows in
  `boot_replicates`, finite standard errors equal to the covariance of the
  finite rows); the masked-dyad guard on `simulate_multi_ergm`.
- **`WithinLayer{T}`** is parameterized on the wrapped term type, so the
  MPLE design and the sampler dispatch statically; the change-statistic fill
  over a `TermSet` is a generated function (`_change_stat_layer_all!`, the
  pattern of `ERGM.change_stat_all!`) — the per-step numbers are under
  *Changed* (the sampler entry).
- Tests: an undirected 5-actor two-layer fixture with a brute-force sweep of
  the undirected-only terms (`Kstar`, `GWDegree`, `Degree`); a
  "Cross-package reach-ins target public bindings" testset that greps the
  source for `ERGM._x`/`Networks._x` and asserts `Base.ispublic` (the two
  boundary helpers adopted ahead of their `public` declaration are
  `@test_broken`, so the allowance fails the day they become public); a
  bit-identity test of `simulate_multi_ergm` against a hand-written
  `ERGM.mh_toggle!` call.
- **Provenanced golden fixture against a real `ergm.multi` fit** (issue #8).
  `test/fixtures/twolayer_ergm_multi.toml` freezes an ergm.multi 0.3.0 fit of
  `Networks(list(net1, net2)) ~ N(~edges + nodematch("grp"), ~factor(.NetworkID) - 1)`
  — per-layer edges and nodematch on a simulated two-layer, 20-actor directed
  network — regenerable with `Rscript test/fixtures/r/twolayer_ergm_multi.R >
  test/fixtures/twolayer_ergm_multi.toml`. Both layers are frozen as edge lists.

  Every term is dyad-independent, so the likelihood factorizes over the
  `(layer, i, j)` dyads and both packages compute the same **exact MLE**.
  ERGMMulti.jl reproduces a tightly-converged (`epsilon = 1e-14`) R `glm` to
  **~1e-13** in the coefficients and **~1e-11** in the standard errors. (ergm's own
  MPLE `glm` stops at the default `epsilon = 1e-8`, leaving it 1.3e-9 / 4.8e-6 from
  the optimum; the fixture measures that and sets the as-shipped tolerances from
  the measurement.)

  **Offsets are validated, and validated against the bug that matters.** A second
  fit pins the two nodematch coefficients at `offset.coef = (0.75, 0.40)` and
  leaves only the edges coefficients free; ERGMMulti.jl's `offsets=Dict(...)`
  reproduces R to ~1e-13. Crucially the fixture *also* freezes the fit of the same
  model with **no nodematch term at all** — which is where a fit lands if the
  offsets are dropped from the linear predictor rather than merely from the
  parameter vector — and the testset asserts the offset fit is nowhere near it
  (−1.418 vs −1.030). A test that only checked "matches R" could pass while both
  sides did nothing; this one cannot.

- **Robust standard errors: `ergm_multi(m, terms; se=:bootstrap)`** (also via
  `fit_multi_ergm`), with the same keywords and semantics as `ERGM.mple`'s:
  `n_boot=100`, `boot_burnin`, `boot_interval`, `rng`. Simulate `n_boot`
  multilayer networks at the fitted coefficients (offsets included) with
  `simulate_multi_ergm`, refit `ergm_multi` on each with the same offsets, and
  report the empirical covariance of the free coefficients — on the ONE shared
  `Networks.bootstrap_cov` loop. **The point estimates are unchanged; only the
  covariance is replaced**, and offset rows stay `NaN` (a fixed coefficient
  carries no uncertainty). Until now the only standard errors available were the
  inverse pseudo-Hessian, anticonservative for any dyad-dependent term over the
  `(layer, i, j)` dyad universe, and printed with significance stars (issue #9,
  ERGMMulti#1). On the test fixture the bootstrap SEs exceed the Hessian ones on
  every coefficient.
- `se_method(fit)` now reports what was actually used (`:hessian`/`:bootstrap`),
  read off the new `MultiERGMResult.se_type` field, and `approximations(fit)`
  and `show` drop the anticonservatism caveat when a bootstrap was used (they
  keep the *point-estimate* MPLE caveat, which holds either way). `show` now
  names the standard-error estimator on its own line.

- Real MPLE estimation: `ergm_multi` builds a design matrix over
  within-layer dyads and fits a logistic pseudo-likelihood via the shared
  `newton_fit` (Newton–Raphson with step halving; `ERGM.newton_fit` when
  this entry was written, `Networks.newton_fit` since it moved), which also
  supplies `vcov` and the standard errors — replacing the placeholder
  fixed-step gradient loop and the hand-rolled local optimizer.
- `WithinLayer(term, layer)` lifts any ERGM.jl term into a layer (analogue
  of `ergm.multi`'s `Layer(~term)`).
- Per-layer fixed coefficients via the `offsets::Dict{Int,Float64}` keyword
  (reported with `NaN` SEs).
- `gof(::MultiERGMResult)` extending the ecosystem-wide `Networks.gof`
  generic, returning a `Networks.GOFResult` with `(1+k)/(N+1)` Monte-Carlo
  p-values.
- StatsAPI accessors on `MultiERGMResult`: `coef`, `stderror`, `vcov`,
  `loglikelihood`, `aic`, `bic`, `nobs`, `dof`.
- New accessors/mutators: `add_layer!`, `add_layer_edge!`, `layer_network`,
  `layer_names`, `add_cross_level_edge!`, `change_stat_layer`;
  `MultiERGMModel`/`MultiERGMResult` are now exported.

### Changed

- **The sampler runs on `ERGM.mh_toggle!`** (item 28), the family's one
  Metropolis kernel, with `(layer, i, j)` moves: `propose` draws the layer,
  then an ordered pair of distinct actors (swapped to `i < j` on undirected
  layers), `change!` fills the generated `_change_stat_layer_all!` and reports
  whether the dyad is currently a tie, `apply!` toggles the layer's edge,
  `on_sample` pushes an attribute-preserving copy. The kernel draws from the
  rng in the same order as the loop it replaced (proposal, then the
  acceptance uniform), so **seeded output is bit-identical** to the
  pre-kernel loop (pinned literally, above) — the draws a user sees change
  only through the new dyad-scaled `burnin`/`interval` defaults (Breaking).
  Per step the sampler allocates nothing of its own: 0 bytes for the
  change-statistic vector at 8 and at 34 statistics, and under 4 bytes per
  step in total (Graphs.jl growing adjacency vectors on accepted additions),
  where the pre-kernel loop's `for (k, t) in enumerate(terms)` over an
  abstractly typed vector cost ~120 bytes per step in dynamic dispatch.
- **Shared helpers replace private copies and reach-ins** (item 13 / 28):
  z → p is `Networks.z_pvalues` (NaN-aware, floored — offset rows print `NaN`
  as before) instead of the private `ERGM._z_pvalues`; the `se=` keyword is
  validated by `Networks.check_se(se, (:hessian, :bootstrap);
  context="ergm_multi")` (message: "ergm_multi: se must be one of (:hessian,
  :bootstrap) (got :sandwich)"); `newton_fit` and `logistic_derivatives` are
  imported from Networks.jl, where they now live; the sampler runs on
  `ERGM.mh_toggle!` with `(layer, i, j)` moves; direction traits are the
  public `ERGM.requires_directed`/`requires_undirected`.
- `show(::MultiERGMResult)` prints `coeftable(fit)` through the shared
  `Networks.print_coeftable` (R-style table with significance codes — so
  the printed table IS the inspected one), names the standard-error
  estimator on its own line, adds the MPLE dyad-dependence caveat where the
  formula has a dyad-dependent term, and the non-convergence (or, for a
  separated fit, R's "MPLE does not exist" verdict), fixed-coefficient and
  bootstrap-exclusion notes; `approximations(fit)` lists the same sentences.
- The "Cross-package reach-ins target public bindings" testset now also
  allows `ERGM._separated` and `ERGM._warn_separated` as `@test_broken`
  pending their `public` declaration in ERGM.jl (requested).
- The "Robust standard errors" testset now asserts that the *dyad-dependent*
  term's bootstrap SE exceeds the Hessian one and that the dyad-independent
  edges SEs agree within a factor of 2, instead of `all(boot .> hess)`: the
  latter held only because boundary replicates used to inflate every column.
- README, docs and CLAUDE.md describe the current behaviour (the
  `using ERGMMulti, Network` typo in four pages is fixed; every snippet
  runs under the ecosystem's `check_snippets.jl`). The estimation guide is
  organised around what a user meets — *Boundary statistics and
  separation*, *Non-convergence*, *Standard errors*, *Missing dyads*,
  *Defaults* — each quoting the exact warning or error sentence; the terms
  guide gains *Validation* (what `WithinLayer` checks against its layer,
  directed-only/undirected-only terms, the `L<l>.gwesp.OTP.fixed.<d>`
  labels) and a *Correspondence with `ergm.multi`* table mapping every term
  to its R term and the fixture that pins it; the index page carries the
  "Not implemented" list (below) and every API page lists every export
  (`fit_ergm_multi`, `layer_names`, the StatsAPI accessors, the deprecated
  `fit_multi_ergm`). The Documenter build is strict.
- `Nestedness` now returns a proportion of within-parent edges (was a raw
  count); `LevelHomophily` counts edges on its own level's network (was
  hard-wired to level 1). Both are documented as descriptive statistics.

### Fixed

- **This changelog contradicted itself on separation under offsets** (round
  3): the Breaking entry above still ended with the round-1 "*Limitation:*
  … the test is skipped when any offset is non-zero" paragraph after the
  round-2 fix below had made it false. The paragraph is gone; the entry
  says "with or without offsets", which is what the code does.
- **The unreachable `:bipartite` drop in `_combine_report` is removed**
  (round 3), together with the `combine_networks` docstring's and
  CLAUDE.md's claim that the report names "a layer's two-mode partition":
  since round 2 `add_layer!` refuses a two-mode layer, so no
  `MultilayerNetwork` can hold one and the branch could never execute.
- **Perfect separation is detected under a non-zero offset** (round 2).
  `_multi_separated` returned `false` as soon as any offset contribution was
  non-zero, so the test suite's own separated design fitted with
  `offsets=Dict(2 => log(4/11))` came back `converged=true,
  separated=false` with coefficients 20.7 / 25.1 and standard errors in the
  tens of thousands, and no warning — violating "an approximate result is
  loud". The offset contribution `η0` is now handed to `ERGM._separated` as
  an extra design column with a fixed coefficient of 1 and standard error 0
  (the derivative closure ignoring that entry), so the linear predictor it
  inspects is the fitted `Xf θ + η0`; the "detected only for models without
  offsets" limitation is gone from CLAUDE.md, README, the docs index, the
  estimation guide and this file.
- The uncalled `_layer_index(m, ::Int)`/`_layer_index(m, ::Symbol)` helpers
  (a leftover of the adapter rewrite; `layer_network(m, ::Symbol)` does the
  lookup inline) are deleted (round 2).
- Sampling working copies preserve attributes: `_copy_net` now delegates to
  the attribute-preserving `Base.copy(::Network)`, so covariate terms no
  longer evaluate to zero during simulation and GOF (part of the
  ecosystem-wide attribute-dropping-copy fix).

### Performance

- **Time to first fit** (item 18; the PrecompileTools workload above).
  Measured in a fresh process (Julia 1.12.6, Linux x86-64, package already
  precompiled), `@time` of each call in turn on a 6-actor two-layer directed
  network with `[LayerEdges(1), LayerEdges(2), WithinLayer(NodeMatch(:grp), 1),
  InterlayerDependence(1, 2)]` — before → after:
  `using ERGMMulti` 0.84 s → 0.83 s (the load is dominated by the
  dependencies); first `ergm_multi` **1.90 s → 0.010 s**;
  `coeftable`/`confint`/`show` 0.35 s → 0.003 s; first `simulate_multi_ergm`
  0.22 s → 0.004 s; first `gof` 0.45 s → 0.027 s. The precompile step itself
  takes ~7 s once per environment.
- **The MPLE derivative loop no longer allocates (review finding 15).**
  `_multi_mple_fit` carried its own logistic loop with a per-dyad
  `(pr*(1-pr)) .* (x * x')` inside it — a fresh `pf×pf` matrix on every one of
  the `n_dyads` rows of every Newton evaluation, **649 KB per evaluation** on a
  2-layer, 40-actor design. The pseudo-likelihood over the within-layer dyads
  *is* a logistic likelihood with an offset, so it now runs on the shared
  `Networks.logistic_derivatives` (hosted in ERGM.jl when this entry was
  written, moved to Networks.jl in the same release): **304 bytes** per evaluation (the gradient and
  Hessian it returns, nothing else), independent of the number of dyads, and
  **7.8x faster** (0.796 ms -> 0.102 ms). Pinned by an `@allocated` regression
  test. `_multi_mple_design` is factored out as the design builder.
  The summation order moves from row-wise accumulation to BLAS, so the
  arithmetic is not bit-identical — but the fitted coefficients are: measured
  against the old loop on the same design, **max|Δθ| = 1.1e-16** (one ulp).
  Newton's last step is quadratically convergent, so a last-ulp difference in
  the gradient and Hessian does not move the fixed point. The golden
  `ergm.multi` fixture agrees at 5.0e-13 (coefficients) and 1.4e-11 (SEs)
  against its 1e-6 exact-MLE tolerance — the same as before.

### Not implemented (known limitations, disclosed)

These are the parts of `ergm.multi` that have no counterpart in 0.2.0. Each
entry states exactly what a user gets instead; the same four items appear
in the README and on the documentation index.

- **No MCMLE.** Estimation is maximum pseudo-likelihood over the
  within-layer dyads, for every model; there is no `method=:mcmle`
  keyword, so for a dyad-dependent multilayer model R's
  `ergm(Layer(...) ~ ...)` default estimator (MCMLE) has no counterpart.
  What you get: an MPLE point estimate, `is_exact(fit) == false`, a
  pseudo-likelihood caveat printed by `show`, `approximations(fit)` naming
  the approximation, and `se = :bootstrap` as the offered remedy for the
  covariance (the point estimates are unchanged by it).
- **No covariate-driven multi-network models.** `ergm.multi`'s
  `Networks()`-style models with per-network coefficients driven by a
  network-level covariate (`N(~edges, ~x)`) are not available;
  `MultiNetwork` is a descriptive container (`CrossNetEdges` only) and
  `ergm_multi` accepts a `MultilayerNetwork` only. What you get: the
  per-layer (`LayerEdges(l)`) and pooled (`LayerEdges()`,
  `LayerEdges([1, 2])`) parameterisations of a multilayer network — every
  layer-specific effect is its own coefficient or an offset, never a
  function of a covariate.
- **Layer logic is limited to same-dyad conjunction and cross-layer
  reciprocity.** `InterlayerDependence(l1, l2)` is `L(~edges, ~A&B)` and
  `MultiplexMutual(l1, l2)` is `mutualL(Ls = list(~A, ~B))`. `twostarL`,
  the `espL`/`dspL`/`gwespL` cross-layer path families, `CMBL`, general
  layer-logic formulas (`~A|B`, `~!A`, `~A&!B`, ...) and bipartite layers
  (`b1dspL`, ...) are not implemented. What you get: no such term exists to
  construct, and a two-mode (bipartite) network is refused by
  `add_layer!`/`as_multilayer` with "layer :a is two-mode (bipartite);
  ERGMMulti models one-mode layers only — the within-layer dyad universe
  would enumerate the impossible within-mode pairs as observed non-ties, so
  the fit is refused instead (statnet's bipartite layer terms b1dspL,
  b2dspL, ... are not implemented)"; a single-layer statistic is expressed
  with `WithinLayer(term, l)` wherever ERGM.jl has the term.
- **Multilevel statistics are descriptive.** `Nestedness`, `LevelHomophily`
  and `CrossLevelEdge` on a `MultilevelNetwork` are evaluated with
  `compute` only and never enter a fit; `ergm_multi(::MultilevelNetwork,
  ...)` is a `MethodError`. What you get: the numbers, and nothing that
  looks like an estimate.

One smaller limitation of the estimator is recorded above where it arises:
the inverse-Hessian standard errors of a dyad-dependent model are
anticonservative (`se=:bootstrap` is the remedy).

## [0.1.0] - 2026-02-09

Initial release: multilayer/multilevel network types and prototype
multilayer ERGM terms and estimation.
