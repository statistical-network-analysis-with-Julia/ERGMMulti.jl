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
  `ERGM.newton_fit`, replacing the placeholder fixed-step gradient loop.
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

- `show(::MultiERGMResult)` prints through the shared
  `Networks.print_coeftable` (R-style coefficient table with significance
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

- **The MPLE derivative loop no longer allocates (review finding 15).**
  `_multi_mple_fit` carried its own logistic loop with a per-dyad
  `(pr*(1-pr)) .* (x * x')` inside it — a fresh `pf×pf` matrix on every one of
  the `n_dyads` rows of every Newton evaluation, **649 KB per evaluation** on a
  2-layer, 40-actor design. The pseudo-likelihood over the within-layer dyads
  *is* a logistic likelihood with an offset, so it now runs on the shared
  `ERGM.logistic_derivatives`: **304 bytes** per evaluation (the gradient and
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
- Optimization, vcov, and SEs delegate to the shared `ERGM.newton_fit`
  (Newton–Raphson with step halving), removing the hand-rolled local
  optimizer.

## [0.1.0] - 2026-02-09

Initial release: multilayer/multilevel network types and prototype
multilayer ERGM terms and estimation.
