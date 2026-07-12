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

- Real MPLE estimation: `ergm_multi` builds a design matrix over
  within-layer dyads and fits a logistic pseudo-likelihood via the shared
  `ERGM.newton_fit`, replacing the placeholder fixed-step gradient loop.
- `WithinLayer(term, layer)` lifts any ERGM.jl term into a layer (analogue
  of `ergm.multi`'s `Layer(~term)`).
- Per-layer fixed coefficients via the `offsets::Dict{Int,Float64}` keyword
  (reported with `NaN` SEs).
- `gof(::MultiERGMResult)` extending the ecosystem-wide `Network.gof`
  generic, returning a `Network.GOFResult` with `(1+k)/(N+1)` Monte-Carlo
  p-values.
- StatsAPI accessors on `MultiERGMResult`: `coef`, `stderror`, `vcov`,
  `loglikelihood`, `aic`, `bic`, `nobs`, `dof`.
- New accessors/mutators: `add_layer!`, `add_layer_edge!`, `layer_network`,
  `layer_names`, `add_cross_level_edge!`, `change_stat_layer`;
  `MultiERGMModel`/`MultiERGMResult` are now exported.

### Changed

- `show(::MultiERGMResult)` prints through the shared
  `Network.print_coeftable` (R-style coefficient table with significance
  codes) and adds an MPLE dyad-dependence caveat where appropriate.
- `Nestedness` now returns a proportion of within-parent edges (was a raw
  count); `LevelHomophily` counts edges on its own level's network (was
  hard-wired to level 1). Both are documented as descriptive statistics.

### Fixed

- Sampling working copies preserve attributes: `_copy_net` now delegates to
  the attribute-preserving `Base.copy(::Network)`, so covariate terms no
  longer evaluate to zero during simulation and GOF (part of the
  ecosystem-wide attribute-dropping-copy fix).

### Performance

- Optimization, vcov, and SEs delegate to the shared `ERGM.newton_fit`
  (Newton–Raphson with step halving), removing the hand-rolled local
  optimizer.

## [0.1.0] - 2026-02-09

Initial release: multilayer/multilevel network types and prototype
multilayer ERGM terms and estimation.
