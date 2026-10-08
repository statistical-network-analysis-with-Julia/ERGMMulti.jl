# Changelog

All notable changes to ERGMMulti.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

First public release of ERGMMulti.jl, a Julia port of R's `ergm.multi`
(statnet) for multilayer networks: several relations on the same actors,
modelled on `ergm.multi`'s block-diagonal construction, fitted by R's rule
(maximum pseudo-likelihood for a dyad-independent formula, Monte-Carlo
maximum likelihood otherwise) over the within-layer dyads, with a
Metropolis sampler, parametric-bootstrap standard errors and goodness of
fit. Statistics, coefficient labels, coefficients and standard errors are
checked against `ergm.multi` 0.3.0 by six provenanced fixtures. The changes
below are relative to 0.1.0, a development version, never released.

**Dependency renamed:** the foundation package is now `NetworkCore` (developed as `Networks`); write `using NetworkCore` where code said `using Networks`. Types and functions keep their names.

### Highlights

- **Any ERGM.jl term within a layer or pooled over layers.**
  `WithinLayer(term, l)` is `L(~term, ~A)`; `WithinLayer(term, [l1, l2])`
  and `WithinLayer(term, :)` pool the term over layers under one
  coefficient, as `L(~term, c(~A, ~B))` does. Attribute terms and expanding
  terms (`NodeFactor`, `NodeMix`, `Degree(0:2)`) work in both forms.
- **R's estimator by default.** `ergm_multi` (also `fit_ergm_multi`) takes
  `method=:auto`, ERGM.jl's `resolve_method` rule: the MPLE over the
  within-layer dyads (offsets supported, on the shared
  `NetworkCore.newton_fit` / `logistic_derivatives` kernel) when no term is
  dyad-dependent, where it is the exact MLE and reproduces R to ~1e-13, and
  Monte-Carlo maximum likelihood otherwise, as R's `ergm.multi` does.
  `show`'s `Method:` line says which estimator ran and why; a keyword of
  the estimator not chosen is an `ArgumentError` naming the `method` that
  takes it.
- **Monte-Carlo maximum likelihood.** Monte-Carlo Newton steps from the MPLE
  through ERGM.jl's shared MCMLE iteration (Hummel's step length, R ergm 4's
  confidence stopping rule with its sample boost), standard errors from the
  Fisher information plus the Monte-Carlo error, the log-likelihood by path
  sampling, offsets held fixed. A fit that does not converge says so in its stopping
  rule's terms (the warning, `show` and `approximations` quote the
  equivalence test's p-value and the step length γ under the default
  `termination=:confidence`; the Hotelling p-value and largest t-ratio only
  under `:hotelling`); `fit.mcmc.convergence` is documented as describing the
  sample drawn before the last step, as in R, and `fit.mcmc` records
  `conv_confidence` and `conv_precision`. It matches `ergm.multi` within R's own
  seed-to-seed spread and exact enumeration on a small network.
- **R's coefficient labels.** `L(friendship)~edges`,
  `L((friendship,advice))~gwesp.fixed.0.5`, `L(friendship&advice)~edges`,
  `L(friendship,advice)~mutual`, `offset(L(advice)~nodematch.g)`, with the
  layers named by `layer_names(m)`, pinned exactly against `ergm.multi`.
- **Honest inference under dependence.** The MPLE of a dyad-dependent
  formula (`method=:mple`) reports estimates and naive standard errors but
  no z values, p-values or confidence intervals; the MCMLE and
  `se=:bootstrap` give calibrated ones, `se=:hessian` the naive Wald table
  on request.
- **Loud failures instead of silent mis-fits.** Formulas are validated
  against the data at construction; boundary statistics, separation,
  non-convergence, masked dyads, two-mode layers and self-loops are refused
  or reported in words.
- **Goodness of fit with auxiliary statistics**: multiplexity and per-layer
  degree and edgewise-shared-partner distributions beside the model
  statistics.

### Breaking

Data model

- `MultilayerNetwork{D}`, `MultiERGMModel{D}` and `MultiERGMResult{D}` carry
  the directedness as a type parameter; layers are `Network{Int,D}` in a
  vector with `layer_names`. Build with `MultilayerNetwork(n; directed)`
  then `add_layer!(m, :name)`; read with `layer_network(m, index_or_name)`.
  `add_layer!` refuses a layer of the other directedness.
- Terms select layers by integer index (`LayerEdges(2)`, `LayerEdges()` for
  all layers), not by `Symbol`. A name inside a term is an `ArgumentError`
  that gives `findfirst(==(:name), layer_names(m))`.
- The term protocol is `change_stat_layer(term, m, l, i, j)`: the
  add-direction, state-independent change statistic.
- `combine_networks(m)` builds the block-diagonal `n × L` network (it no
  longer set-combines networks); `as_multilayer(nets, names)` and
  `split_by_layer(combined, n, L; names)` take vectors.
- `simulate_multi_ergm(m, terms, θ; n_sim, burnin, interval, rng)` returns a
  vector of `MultilayerNetwork`s.
- `ergm_multi` accepts a `MultilayerNetwork` only. The development-era
  alias `fit_multi_ergm` is removed (use `fit_ergm_multi`); the docs list
  every removed name.
- `MultiERGMResult` has four more fields, `method`, `mcmc`,
  `separated_terms` and `inference_withheld` (positional layout changed).
- Removed exports: `LayerSpec`, `LevelSpec`, `LayerLogic`, `BetweenLayers`.
- Coefficient labels are `ergm.multi`'s, with the layers named by
  `layer_names(m)`: `L(A)~edges`, `L((A,B))~edges` (pooled), `L(A)~mutual`,
  `L(A)~triangle`, `L(A&B)~edges` (`InterlayerDependence`, was
  `duplex.1.2`), `L(A,B)~mutual` (`MultiplexMutual`, was
  `duplex.mutual.1.2`), `L(A)~<term>` (`WithinLayer`, direction-aware:
  `L(A)~gwesp.OTP.fixed.0.5` on a directed layer), and `offset(<label>)`
  for an offset row (was `<label> (offset)`). Code that looked a
  coefficient up by its old label must use the new one.
- **The default estimator is R's**: `method=:auto` fits the MCMLE for a
  dyad-dependent formula, where the MPLE was the default. Pass
  `method=:mple` for the MPLE; `se=`, `n_boot` and the other MPLE keywords
  on a dyad-dependent formula without it are an `ArgumentError`.
- Minimum Julia is 1.12; the package UUID changed.

Estimation and inference

- **The MPLE of a dyad-dependent formula withholds naive inference.**
  `se` defaults to `nothing`: the standard errors are the inverse
  pseudo-Hessian's, but `coeftable`/`show` report `NaN` z and p-values,
  `confint` throws an `ArgumentError`, `fit.inference_withheld` is `true`
  and `approximations(fit)` records it. The naive errors under-cover (95%
  Wald intervals covered 0.76–0.94 in simulation; the bootstrap 0.90–0.96).
  Pass `se=:hessian` for the naive Wald table or `se=:bootstrap` for
  calibrated inference. Dyad-independent fits are unchanged.
  `MultiERGMResult` has fifteen fields and one positional constructor,
  which checks `separated` against `separated_terms`.
- A statistic at the boundary of its attainable range gets R ergm's `drop`
  semantics in both estimators: its coefficient is fixed at `∓Inf`
  (standard error 0, p-value 0) with a warning, and the rest are estimated
  with it held at its bound — the MPLE on the dyads it does not touch, the
  MCMLE (the default for a dyad-dependent formula, which used to refuse
  such a model) with a sampler that never moves it off its observed value.
  The bound is read off the statistic's attainable range as well as the
  pseudo-likelihood design, so a term whose change statistics are all 0
  (`LayerTriangle` on a layer with no two-path) is caught: its MPLE used to
  stop at 0 for every coefficient. `drop=false` refuses such a model
  instead. When the held statistic is dyad-dependent the MCMLE's
  log-likelihood, AIC and BIC are `NaN`, with the reason in `show` and
  `approximations`. `is_exact(fit)` is `false`. `gof` simulates such a fit
  with the statistic held at its bound; `se=:bootstrap` and
  `simulate_multi_ergm` refuse a non-finite coefficient. `ergm.multi` 0.3.0
  itself reports `NA` or a finite value where its estimation stopped; its
  offset spelling (`offset(L(...))` at `-Inf`) fits the model the drop
  fits, and the fixture `boundary_multi.toml` pins the two together.
- A statistic with no identifiable coefficient (its change statistics all
  0 on the dyads fitted, or a linear combination of the others, such as two
  copies of one term) is reported as `NaN`, R's `NA`, with a warning, and
  the others are fitted without it; Newton used to stop at 0 for every
  coefficient and report non-convergence. `se=:bootstrap` refuses such a
  fit; the MCMLE holds a dyad-independent one at 0 and needs `init=` for a
  dyad-dependent one.
- Separation by a combination of statistics is detected by NetworkCore's
  shared separation verdict on the design actually fitted, with or without
  offsets, and handled by the ecosystem's policy: a warning naming the
  separating terms, `converged == false`, `separated == true`, the terms in
  `fit.separated_terms`, and `NaN` z values, p-values and confidence
  intervals. `se=:bootstrap` and an MCMLE start are refused on such a fit.
- `MultiERGMModel` validates the formula before any fitting: layer indices,
  every `WithinLayer` term against every layer it selects (ERGM.jl's own
  checks for attributes, covariate sizes and direction requirements), bare
  ERGM.jl terms and multilevel descriptives, self-loops, networks without a
  within-layer dyad, and offsets. `LayerMutual` and `MultiplexMutual` are
  refused on undirected layers instead of returning 0.
- Expanding within-layer terms contribute one coefficient per statistic
  (`NodeFactor` levels, `Degree(0:2)`); an offset on an expanding term is
  refused.
- Masked (missing) dyads are refused by `ergm_multi` and
  `simulate_multi_ergm`, naming the layer; there is no `missing=` keyword.
  Two-mode layers are refused by `add_layer!`.
- `add_layer_edge!` refuses an actor outside `1:n` and a self-loop on a
  `loops=false` layer; `layer_network(m, l)` out of range is an
  `ArgumentError`.
- Sampler defaults are dyad-scaled (`burnin = 20 × n_dyads`,
  `interval = max(100, n_dyads ÷ 10)` over the within-layer dyads).
- `gof` returns more panels (see Added); code that indexed
  `g.statistics[2]` for the layer edges is unaffected, code that assumed two
  panels is not.

### Added

- **Pooled within-layer terms**: `WithinLayer(term, layers)` with a vector
  of layer indices or `:`. The statistic is the sum of the term's per-layer
  statistics (`ergm.multi`'s `L(~term, c(~A, ~B))`), labelled
  `L((A,B))~<term>`. Each selected layer is validated and its
  attributes are snapshotted separately. Refused with an `ArgumentError`:
  an empty or repeating layer vector, a pooled expanding term whose
  statistics differ between layers, and `WithinLayer(Offset(...), l)`.
  Before, `WithinLayer(GWESP(0.5), [1, 2])` was a `MethodError`.
- **Fixture `multilayer_labels.toml`** (`ergm.multi` 0.3.0): the coefficient
  labels R prints for every multilayer term on three named directed layers
  and two undirected ones — pools over a subset, over every layer and over
  one layer, both orders of a layer pair, `mutualL`, attribute and expanding
  terms, and `offset(...)` — compared exactly, with the statistics at 1e-10.
- `multilayer_mcmle.toml` also freezes R's MPLE refit at `glm` epsilon
  1e-14; the as-shipped MPLE is compared at the fixture's tolerance (20× the
  measured `glm` slack) and the exact refit at 1e-6. The as-shipped
  tolerances of the four MPLE fixtures have no floor: each is 20× its
  measured slack, rounded up to a power of ten (1e-7 or 1e-8 where 1e-6
  stood).
- **Fixture `boundary_multi.toml`** (`ergm.multi` 0.3.0): three designs with
  a statistic at its bound (a triangle term with all-zero change
  statistics, a cross-layer term, a dyad-independent `nodematch`). It pins
  what `ergm.multi` reports at its defaults, the MPLE's exact limit (1e-8)
  and the MLE with the statistic held, fitted by R as an offset at `-Inf`
  under ten seeds; the default fit must lie within four of R's
  seed-to-seed standard deviations.
- **Fixture `pooled_layer_terms.toml`** (`ergm.multi` 0.3.0): `summary()` of
  nineteen terms pooled over two directed layers and fifteen over two
  undirected layers (1e-10, R's labels), and two dyad-dependent MPLE fits
  with pooled terms against R's design refit at `glm` epsilon 1e-14 (1e-6).
- **`gof` auxiliary panels**: `"multiplexity"` (dyads tied in exactly `k`
  layers) and, per layer, `"degree: <layer>"` (or `"idegree"`/`"odegree"`)
  and `"esp: <layer>"`.
- `se=:bootstrap`: parametric bootstrap on the shared
  `NetworkCore.bootstrap_cov` loop (`n_boot`, `boot_burnin`, `boot_interval`,
  `rng`, `threaded`); results do not depend on the thread count.
  Replicates without a finite MPLE are excluded, kept as `NaN` rows of
  `fit.boot_replicates` and warned about once; the warning, `show` and
  `approximations(fit)` carry the ERGM family's sentence "The standard
  errors are conditional on a finite refit: the excluded replicates are the
  extreme ones, so the standard errors are biased downward."
- `fit_ergm_multi` (the ecosystem's `fit_<model>` name; the same function
  as `ergm_multi`), the validating `MultiERGMModel(terms, m; offsets)`
  constructor, `name(term, m)`, `has_dyad_dependent(model)`.
- The full StatsAPI surface on `MultiERGMResult` (`coef`, `coefnames`,
  `stderror`, `vcov`, `confint`, `loglikelihood`, `nobs`, `dof`, `aic`,
  `bic`, `coeftable`) and the result metadata (`is_exact`, `se_method`,
  `approximations`, ...). `coefnames(fit)` (StatsAPI) returns the
  coefficient labels.
- The adapters follow the conversion contract: `combine_networks` and
  `split_by_layer` preserve the missing-dyad mask and report what they drop
  (`report=true` returns a `ConversionReport`).
- `Base.show` for every public type; a docstring with a runnable example
  for every export (executed by the test suite).
- Fixtures `twolayer_ergm_multi.toml` (exact MLE with and without offsets)
  and `twolayer_layer_terms.toml` (every layer term's statistic, a
  dyad-dependent MPLE, the boundary design and undirected layers).
- A precompile workload: the first `fit_ergm_multi` of a session takes
  about 0.01 s instead of 1.9 s.

### Changed

- The sampler runs on the ERGM family's Metropolis kernel
  (`ERGM.mh_toggle!`) with `(layer, i, j)` moves; simulated networks keep
  their vertex attributes.
- The Newton optimizer, the logistic derivatives, the z → p map, the
  boundary test, the separation verdict and the bootstrap loop are the
  shared NetworkCore.jl / ERGM.jl implementations; the MPLE runs on ERGM.jl's
  pseudo-likelihood fitter (`ERGM.Extension.mple_fit_design`), the one every
  ERGM-family MPLE uses.
- **Built on ERGM.jl's extension API** (`ERGM.Extension`): the layer terms
  declare their attainable ranges as methods of
  `ERGM.Extension.attainable_range` on a `MultilayerNetwork` (a pooled
  within-layer term's range is the sum of ERGM.jl's per-layer ranges), so
  no range table and no non-public reach-in remain. The MPLE's warnings are
  the fitter's own, worded for ergm.multi through its `note=` keyword, and
  its separation verdict is the one the fit returns, so it is computed once
  per fit. Two warning sentences now read as ERGM.jl's do: "do not vary"
  adds "the observed network may sit at an extreme point of the sample
  space", and the linear-dependence warning says "the other coefficients
  are the MPLE without them".
- CI and documentation workflows rebuild the sibling layout from
  `[sources]`; `docs/Project.toml` carries `[compat]`.

### Performance

- One MPLE derivative evaluation allocates 304 bytes regardless of the
  number of dyads (was 649 KB on a 2-layer, 40-actor design) and is 7.8×
  faster; the fitted coefficients are unchanged to one ulp.
- The change-statistic fill is allocation-free per Metropolis step,
  including pooled attribute terms.

### Known limitations

The same items appear in the README and on the documentation index ("Not
implemented").

- **The MCMLE is not R's implementation.** `ergm_multi` follows R's
  default (`method = :auto`): the MPLE, which is the exact MLE, when every
  term is dyad-independent, and Monte-Carlo maximum likelihood otherwise
  (`method = :mcmle`; checked against `ergm.multi` within R's seed-to-seed
  spread). The MCMLE uses the package's uniform (layer, i, j) proposal, not
  R's tie/no-tie proposals or `control.ergm` tuning, so R seeds do not carry
  over; it has no missing-data form (a masked layer is refused) and no MCMC
  diagnostics plot. A statistic at the boundary of its attainable range is
  fixed at `∓Inf` by both estimators (R ergm's `drop=TRUE`; `drop = false`
  refuses instead), where `ergm.multi` 0.3.0 reports `NA` or a finite value
  where its estimation stopped; when that statistic is dyad-dependent the
  MCMLE's log-likelihood, AIC and BIC are `NaN`. The MCMLE is refused when
  the MPLE start is separated (an `ArgumentError` naming the terms, where
  `ergm.multi` warns and returns a finite estimate with a huge standard
  error).
  The MPLE of a dyad-dependent formula (`method = :mple`) reports no z
  values, p-values or confidence intervals by default (`NaN` in `coeftable`,
  `confint` throws an `ArgumentError`), because its naive pseudo-likelihood
  standard errors under-cover (95% Wald intervals covered 0.76–0.94 in
  simulation); `se = :bootstrap` gives parametric-bootstrap standard errors
  (coverage 0.90–0.96), which are conditional on a finite refit and biased
  downward when replicates are dropped (counted in `show` and
  `approximations(fit)`), and `se = :hessian` is the written opt-in to the
  naive Wald table R prints for an MPLE fit.
- **No models over several networks.** `ergm.multi`'s `Networks()`
  construction — one model fitted to a sample of networks that may differ in
  size and composition, with `N(~edges, ~x)` network-level covariates and
  the `gofN` diagnostics — is not implemented. `MultiNetwork` is a
  descriptive container (`CrossNetEdges` only) and `ergm_multi` accepts a
  `MultilayerNetwork` only: layers on the *same* actors. What you get: the
  per-layer (`LayerEdges(l)`, `WithinLayer(term, l)`) and pooled
  (`LayerEdges()`, `WithinLayer(term, [1, 2])`) parameterisations of a
  multilayer network — every layer-specific effect is its own coefficient
  or an offset, never a function of a covariate.
- **Layer logic is limited to same-dyad conjunction of edges and cross-layer
  reciprocity.** `InterlayerDependence(l1, l2)` is `L(~edges, ~A&B)` and
  `MultiplexMutual(l1, l2)` is `mutualL(Ls = list(~A, ~B))`. Any ERGM.jl
  term can be evaluated within one layer (`WithinLayer(term, l)`,
  `L(~term, ~A)`) or pooled over layers (`WithinLayer(term, [1, 2])`,
  `L(~term, c(~A, ~B))`), but not on a *logical* layer: `L(~term, ~A&B)` for
  a term other than `edges`, `~A|B`, `~!A`, `~A&!B`, weighted layer sums,
  `twostarL`, the `espL`/`dspL`/`gwespL` cross-layer path families and
  `CMBL` are not implemented. Neither are `Layer(nw, c("attr1", "attr2"))`
  (layers read from edge attributes of one network — build one `Network`
  per layer), the `.symmetric` mixing of directed and undirected layers
  (`add_layer!` refuses a layer of the other directedness) or bipartite
  layers (`b1dspL`, ...). What you get: no such term exists to construct; a
  two-mode (bipartite) network is refused by `add_layer!`/`as_multilayer`
  with "layer :a is two-mode (bipartite); ERGMMulti models one-mode layers
  only — the within-layer dyad universe would enumerate the impossible
  within-mode pairs as observed non-ties, so the fit is refused instead
  (statnet's bipartite layer terms b1dspL, b2dspL, ... are not
  implemented)"; a pooled term that expands into different statistics on
  different layers (a `NodeFactor` whose levels differ between layers) and a
  `WithinLayer(Offset(...), l)` are `ArgumentError`s (use the `offsets=`
  keyword for the latter).
- **Multilevel statistics are descriptive.** `Nestedness`, `LevelHomophily`
  and `CrossLevelEdge` on a `MultilevelNetwork` are evaluated with
  `compute` only and never enter a fit; `ergm_multi(::MultilevelNetwork,
  ...)` is a `MethodError`. What you get: the numbers, and nothing that
  looks like an estimate.
- **Goodness of fit has no geodesic-distance panel.** `gof` compares the
  model statistics, the layer edge counts, the multiplexity distribution
  and each layer's degree and edgewise-shared-partner distributions;
  `gof.ergm`'s geodesic distances and the dyad-wise shared partners are not
  computed.

## [0.1.0]

Development version, never released.
