# Model Estimation

[`ergm_multi`](@ref) (= [`fit_ergm_multi`](@ref), one `const` function:
the R name and the ecosystem's `fit_<model>` name) has two estimators over
the **within-layer dyads**, and by default chooses between them as R's
`ergm.multi` does.

## The default: R's rule

`method = :auto` (the default) fits the MPLE when every term is
dyad-independent — there the pseudo-likelihood *is* the likelihood and the
MPLE is the exact MLE — and Monte-Carlo maximum likelihood (MCMLE, see
[Maximum likelihood (MCMLE)](@ref) below) as soon as one term is
dyad-dependent (`LayerMutual`, `LayerTriangle`, `InterlayerDependence`,
`MultiplexMutual`, a `WithinLayer` of a dyad-dependent ERGM.jl term). The
rule is ERGM.jl's `ERGM.resolve_method`, shared by every fitter of the ERGM
family, and the predicate is `ERGM.has_dyad_dependent(model)`.
`method = :mple` and `method = :mcmle` choose an estimator explicitly. The
`Method:` line of `show` says which ran and why.

Each estimator takes its own keywords: `se`, `n_boot`, `boot_burnin`,
`boot_interval` and `threaded` are the MPLE's; `n_samples`, `burnin`,
`interval`, `mcmle_maxiter`, `termination`, `conv_precision`,
`conv_confidence`, `conv_threshold`, `hotelling_alpha`, `max_n_samples`,
`bridge_rungs`, `bridge_samples` and `init` are the MCMLE's; `maxiter`, `tol`
and `rng` are both. A keyword of the estimator not chosen is refused in
words — typically `se = :bootstrap` on a dyad-dependent formula:

> ergm_multi: keyword `se` is not accepted by method=:mcmle. method=:auto
> chose :mcmle because the formula is dyad-dependent (R's ergm.multi fits
> the Monte-Carlo MLE there). It is a keyword of method=:mple; pass
> method=:mple explicitly to use it.

## The MPLE

`method = :mple` maximizes the pseudo-likelihood: each (layer, i, j)
contributes a logistic term in `θ'Δg` with `Δg` from
[`change_stat_layer`](@ref). It is maximized with the shared
`NetworkCore.newton_fit` optimizer (Newton-Raphson with step-halving) on the
shared, allocation-free `NetworkCore.logistic_derivatives` kernel — the same
code ERGM.jl's MPLE, TERGM.jl's CMPLE and ERGMRank.jl's swap MPLE run on.
Standard errors come from the inverse observed information of the
pseudo-likelihood, or from a parametric bootstrap (`se=:bootstrap`).

```julia
using ERGMMulti, NetworkCore, Random

rng = Xoshiro(1)
m = MultilayerNetwork(20; directed = true)
add_layer!(m, :friendship)
add_layer!(m, :advice)
for i in 1:20, j in 1:20
    i == j && continue
    rand(rng) < 0.1 && add_layer_edge!(m, :friendship, i, j)
    rand(rng) < 0.1 && add_layer_edge!(m, :advice, i, j)
end
terms = [LayerEdges(), InterlayerDependence(1, 2)]

result = fit_ergm_multi(m, terms; method = :mple)
coef(result)
stderror(result)
loglikelihood(result)      # maximized pseudo-log-likelihood
aic(result), bic(result)
nobs(result), dof(result)  # 760 within-layer dyads, 2 estimated coefficients
coeftable(result)          # the R-style table `show` prints
is_exact(result)           # false — InterlayerDependence is dyad-dependent
approximations(result)     # says so, in words
result.inference_withheld  # true — no z, p or interval by default (see Standard errors)
```

`maxiter` (default 100) and `tol` (default 1e-8) control the Newton
iteration; `se`, `n_boot`, `boot_burnin`, `boot_interval`, `rng` and
`threaded` control the bootstrap (below). The keyword vocabulary is the
ERGM family's (`maxiter`, `n_sim`, `rng`), so a call written for
`fit_ergm` reads the same here.

## Validation before fitting

The [`MultiERGMModel`](@ref) constructor (which both `ergm_multi` and
`simulate_multi_ergm` go through) checks the formula against the data and
throws an `ArgumentError` naming the fix — before any fitting:

- a layer index the network does not have (`LayerEdges(3)` on two layers)
  names the layer and `layer_names(m)`;
- a `WithinLayer(term, l)` is validated against layer `l` with ERGM.jl's own
  validator: a vertex attribute the layer lacks (`WithinLayer(NodeMatch(:welth), 1)`),
  an attribute set on only some vertices (statnet refuses NA), an `EdgeCov`
  of the wrong size, an undirected-only term (`Kstar`, `GWDegree`, `Degree`)
  on a directed layer, a directed-only one (`Mutual`, `OStar`, ...) on an
  undirected layer;
- `LayerMutual` and `MultiplexMutual` are directed-only;
- a bare ERGM.jl term (`Mutual()` where `WithinLayer(Mutual(), l)` was
  meant — ergm.multi's `L(~mutual, ~A)`) or a multilevel descriptive
  (`Nestedness(1)`) is refused with the `WithinLayer` fix spelled out ("term
  'mutual' (Mutual) has no multilayer change statistic … lift an ERGM.jl term
  into a layer with `WithinLayer(Mutual(...), l)`"), not left to die inside
  the design builder with a `MethodError`;
- a layer that **contains a self-loop** is refused, as `ERGM.ERGMModel`
  refuses it ("layer :b (layer 2) contains 1 self-loop (at vertex 1).
  ERGMMulti models the off-diagonal dyads of every layer only … R ergm warns
  \"This network contains loops\" here"): the statistics would count the
  loop while the within-layer design, `nobs`, the sampler and `gof` never
  touch the diagonal. A layer built with `loops=true` that holds no loop is
  accepted;
- `offsets` must index terms, be finite, leave at least one coefficient
  free, and not target a term that expands into several statistics;
- the network must have at least one layer and **at least two actors**: a
  `MultilayerNetwork(1)` (or `(0)`) has no within-layer dyad, and the
  constructor says so ("the network has no within-layer dyads (n = 1 actor,
  1 layer); a multilayer ERGM needs at least two actors — nothing to
  estimate or simulate") instead of running Newton on an empty design.

Three checks happen even earlier. **Data entry** is loud:
[`add_layer_edge!`](@ref) refuses an actor id outside `1:n` ("add_layer_edge!:
actor ids must lie in 1:10 (got (0, 2)); the multilayer network has 10
actors. Ids are 1-based — a 0 usually means 0-based data — and an edge is
never dropped silently.") and a self-loop on a layer built without
`loops=true`, and [`layer_network`](@ref) with an out-of-range index names
the layers the network has ("layer index 5 out of range: the network has 2
layers (:friendship, :advice); layer indices must lie in 1:2") — a mistyped
id can no longer leave you with a silently smaller network. A **two-mode
(bipartite) layer** is refused
by `add_layer!`/`as_multilayer` ("layer :a is two-mode (bipartite);
ERGMMulti models one-mode layers only — the within-layer dyad universe
would enumerate the impossible within-mode pairs as observed non-ties …"):
it never enters a `MultilayerNetwork`. And a term that selects a layer by
**name** (`LayerEdges(:advice)`, `WithinLayer(Triangle(), :advice)`,
`InterlayerDependence(:friendship, :advice)`) is an `ArgumentError` at the
term's construction, giving the index to use ("use LayerEdges(k) with
k = findfirst(==(:advice), layer_names(m))").

The full list, with the sentences a user reads, is in [Terms](@ref terms-guide)
under *Validation*.

## Offsets

`offsets = Dict(k => c)` fixes term `k`'s coefficient at `c` — the
`ergm.multi` offset mechanism used for per-layer size adjustments or
theory-fixed effects. Offset terms report their fixed coefficient with
`NaN` standard error (and a `NaN` confidence interval, `NaN` rows of
`vcov`); the free coefficients are estimated with the offset contribution
absorbed into the linear predictor, and `dof` excludes the offsets.
Offsets index the terms as given; a within-layer term that expands into
several statistics (a multi-level `NodeFactor`, a `Degree(0:2)`) cannot be
offset as a whole — pin each level or degree with its own single-statistic
term. The golden fixture checks the offset fit against `ergm.multi`'s
`offset.coef` *and* against the fit with the term dropped, which an offset
fit must not reproduce.

## Boundary statistics and separation

A statistic at the boundary of its attainable range — no co-occurring tie
under `InterlayerDependence`, no mutual dyad under `LayerMutual`, no
triangle under `LayerTriangle`, a `NodeMatch` with no within-group tie in
its layer — has no finite maximum pseudo-likelihood estimate, and no finite
MLE either. Both estimators apply R ergm's default `drop=TRUE` semantics.
The bound is found two ways: from the statistic's attainable range (its
observed value equals the smallest or largest value it can take — the only
way to see it when every change statistic of the term is 0, as for
`LayerTriangle` on a layer with no two-path, where Newton used to stop at 0
for every coefficient), and from the pseudo-likelihood design (one-signed
change statistics), iterated on the dyads the dropped statistics leave.

`ergm.multi` 0.3.0 does **not** drop: its layer operator does not
propagate the statistic's attainable range to `ergm.checkextreme.model`,
so for the same design R reports `NA` (a statistic whose change statistics
are all zero) or warns "The MPLE does not exist!" and returns a *finite*
value (≈ −17.6 with a standard error in the thousands on the design of
section (iii) of `test/fixtures/twolayer_layer_terms.toml`); its MCMLE
keeps that value. Written out as an offset — `offset(L(~edges, ~A&B))`
with `offset.coef = -Inf` — `ergm.multi` fits the model the drop fits
here, and `test/fixtures/boundary_multi.toml` pins ERGMMulti.jl's default
fits against it on three designs (a triangle term with all-zero change
statistics, a cross-layer term, a dyad-independent `nodematch`): the MPLE at
its exact limit, the MCMLE within R's seed-to-seed spread. The MPLE's
warning is R's sentence:

> ergm_multi: observed statistic(s) L(friendship&advice)~edges are at their
> smallest attainable values. Their coefficients will be fixed at -Inf (no finite
> maximum pseudo-likelihood estimate exists; R ergm's drop=TRUE;
> ergm.multi 0.3.0 warns "The MPLE does not exist!" and returns a finite
> value with a standard error in the thousands instead — see the
> estimation guide). The remaining coefficients are estimated on the dyads
> these statistics do not touch — the exact limit of the pseudo-likelihood
> — with standard error 0 and p-value 0 recorded for the fixed ones.

("largest attainable values … fixed at +Inf" for a statistic at its
maximum.) The MCMLE's says the same of the likelihood:

> ergm_multi: observed statistic(s) L(friendship&advice)~edges are at their
> smallest attainable values. Their coefficients will be fixed at -Inf (no
> finite maximum-likelihood estimate exists; R ergm's drop=TRUE — ergm.multi
> 0.3.0 does not drop and reports a finite value where its estimation
> stopped). The remaining coefficients are estimated with these held fixed:
> the sampler never moves the statistic off its observed bound. Pass
> drop=false to refuse such a model instead.

What the result then holds:

- `coef(fit)[k] == -Inf` (or `+Inf`), `stderror(fit)[k] == 0.0`, z
  `∓Inf`, p-value `0` — the `coeftable` row is printed as such;
- the other coefficients are estimated with the statistic held at its
  bound: the MPLE on the dyads the dropped statistic does not touch (the
  exact limit of the pseudo-likelihood as the coefficient goes to `∓Inf`;
  `bic` uses that dyad count as its sample size), or the MCMLE with a
  sampler that rejects every move off the bound (the coefficient is
  `±1e300` in the chain; `fit.mcmc.estimated` lists the coefficients the
  Monte-Carlo iteration estimated); `dof` counts only the finite
  coefficients;
- if the held statistic is dyad-dependent, the MCMLE's log-likelihood, AIC
  and BIC are `NaN`: the path sampler starts from the dyad-independent part
  of the model, which cannot hold such a statistic at its bound (R reports
  a value there);
- `fit.converged` is `true` (the reduced problem converged),
  `is_exact(fit)` is `false`, and `show(fit)`/`approximations(fit)` carry
  the note "L(friendship&advice)~edges fixed at -Inf (observed statistic at its smallest
  attainable value): no finite maximum pseudo-likelihood estimate exists …";
- `se=:bootstrap` is refused on such a fit with an `ArgumentError`
  ("se=:bootstrap is not available when a coefficient is fixed at ±Inf
  …"), and so is `simulate_multi_ergm(fit.model, coef(fit))`: a network
  cannot be simulated at an infinite coefficient;
- `gof(fit)` simulates the fitted model: the chain starts at the observed
  network, which is at the bound, and never leaves it.

`drop = false` (R's `control.ergm(drop = FALSE)`, whose "MLE is poorly
defined", is not implemented) refuses such a model with an `ArgumentError`
before any fit. The MCMLE also refuses statistics fixed at both `-Inf` and
`+Inf`, which its sampler cannot hold together.

A statistic with no identifiable coefficient — every change statistic 0 on
the dyads fitted and no attainable bound to compare with (an all-zero
covariate), or a linear combination of the statistics before it (two copies
of one statistic) — is reported as `NaN`, R's `NA`, with R's warning ("do
not vary on the dyads fitted" or "are linear combinations of the preceding
statistics"), and the other coefficients are fitted without it. The MCMLE
holds a dyad-independent one at 0 and refuses a dyad-dependent one unless
`init=` is given.

```julia
using ERGMMulti, NetworkCore
m = MultilayerNetwork(30; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
add_layer_edge!(m, :friendship, 1, 2)          # one tie, no co-occurrence
fit = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)]; method = :mple)  # warns
coef(fit)[2] == -Inf                # true
stderror(fit)[2] == 0.0             # true
coef(fit)[1] ≈ log(1 / 1738)        # true: the MPLE on the 1739 untouched dyads
dof(fit)                            # 1
is_exact(fit)                       # false
```

Separation by a *combination* of statistics — no single column at its
boundary, yet a linear combination of them perfectly predicts the ties —
is R's "The MPLE does not exist!". It is detected by the ecosystem's one
separation verdict, `NetworkCore.logistic_separation`: a linear programme
(as R ergm's `mple.existence`) whose direction is certified in exact
arithmetic, run on the design actually fitted — the free columns that
survive the boundary drop, on the rows they are estimated on. A finite
offset shifts the linear predictor and cannot change the verdict, so
offset fits are judged alike. The fit then follows the ecosystem's
separation policy. The warning names the separating terms:

> ergm_multi: the MPLE does not exist (separation). The log-likelihood keeps
> increasing as the coefficient(s) on `L(a)~nodecov.x`, `L(a)~nodecov.z` run
> to ±Inf along a direction that predicts … perfectly, so the returned values
> are where the optimizer stopped, not estimates. The fit is reported with
> `converged == false`, the separated terms are flagged, and z values,
> p-values and confidence intervals are withheld (NaN). … R ergm.multi warns
> "The MPLE does not exist!" for the same design.

and the fit comes back with `fit.converged == false`,
`fit.separated == true`, `fit.separated_terms` naming the terms,
`is_exact(fit) == false`, `NaN` z values and p-values in `coeftable`, a
`NaN` [`confint`](@ref), and the verdict printed by `show` (under
`converged: false`) and listed by `approximations(fit)` instead of the
generic non-convergence sentence. `se = :bootstrap` is refused on such a
fit, and so is an MCMLE started from it. (`ergm.multi` itself warns and
returns finite coefficients for the same design.)

## Non-convergence

A Newton iteration that runs out of `maxiter` is loud: the warning

> ergm_multi: the Newton iteration did not converge within maxiter = 100
> iterations (the maximum pseudo-likelihood estimate may not exist: a
> statistic at the boundary of its attainable range, or
> separation); the result reports converged = false and its point
> estimates and standard errors are unreliable

is emitted once, `fit.converged == false`, `is_exact(fit) == false`,
`show(fit)` prints the caveat right under `converged: false`, and
`approximations(fit)` lists it as its first entry. (A rank-deficient design —
two copies of the same statistic — is not a convergence failure: the copy is
reported as `NaN`, above.)
An unconverged fit is never returned as a fit with a footnote: the
warning, the field, the printout and the metadata all agree.

## Standard errors of the MPLE

These are the options of `method = :mple` (the MCMLE reports the inverse
Fisher information plus the Monte-Carlo error; see below).

- **Default (`se` not given)**: the inverse negative pseudo-Hessian at the
  optimum. With only dyad-independent terms (`LayerEdges`,
  `WithinLayer(NodeMatch(...))`, ...) the pseudo-likelihood *is* the
  likelihood, these are the usual maximum-likelihood standard errors (the
  golden fixture holds them to ~1e-11 against R) and the full Wald table is
  reported. With any dyad-dependent term (`LayerMutual`, `LayerTriangle`,
  `InterlayerDependence`, `MultiplexMutual`, a `WithinLayer(GWESP(...))`)
  the (layer, i, j) conditionals are multiplied as if independent and the
  naive standard errors **under-cover**: in two 400-replicate simulations
  of a five-term model (20 actors, two directed layers; data simulated at
  `θ = (−2.6, −2.8, 1.5, 1.2, 0.4)` from two seeds), 95% Wald intervals
  covered 0.76–0.94, the `WithinLayer(GWESP(0.5), 2)` coefficient lowest
  (0.76 and 0.79). The fit therefore
  reports the estimates and the naive standard errors but **withholds the
  inference built on them** — z and p are `NaN` in `coeftable`/`show`,
  [`confint`](@ref) throws an `ArgumentError`, `fit.inference_withheld` is
  `true`, and `show` and `approximations(fit)` say why ("z values, p-values
  and confidence intervals withheld: …"). This is the default of ERGM.jl's
  `mple` too. The MCMLE — what `method = :auto` fits for such a formula —
  reports calibrated standard errors with a full table.
- **`se=:hessian`**: the same standard errors with the naive Wald table (z,
  p, `confint`) — the written opt-in to what R prints for an MPLE fit.
  `show` keeps the warning, and `approximations(fit)` lists
  "inverse-Hessian standard errors of the naive pseudo-likelihood: expected
  anticonservative under dependence".
- **`se=:bootstrap`**: a parametric bootstrap on the ONE shared
  `NetworkCore.bootstrap_cov` loop. `n_boot` (default 100) multilayer networks
  are simulated from the fitted model at `θ̂` (offsets included) with
  [`simulate_multi_ergm`](@ref) — `boot_burnin`/`boot_interval` default to
  the dyad-scaled rule below, `rng` seeds the draws — each is refit with the
  same offsets, and the empirical covariance of the refitted free
  coefficients replaces `vcov`. **The point estimates are unchanged** (and
  remain pseudo-likelihood estimates, biased in finite samples — `show`
  keeps that note); `se_method(fit)` reports `:bootstrap`; offset rows stay
  `NaN`. In the same simulations the bootstrap intervals covered
  0.90–0.96, and the intervals of `method=:mcmle` (Fisher + Monte-Carlo
  standard errors) 0.95–0.99 over the fits that converged at the defaults
  (695 of 761; the other 66 did not converge or stopped with an error at
  the defaults and are excluded).
- **Replicate exclusion.** A replicate whose refit has no finite MPLE (a
  simulated network with no triangle under `LayerTriangle`, no
  co-occurring tie under `InterlayerDependence`, a separated design) is
  excluded from the covariance and kept as a `NaN` row of
  `fit.boot_replicates` (an `n_boot × p_free` matrix; `nothing` under
  `se=:hessian`). One warning reports "k of the n_boot bootstrap refits did
  not converge … and were excluded", `show`/`approximations(fit)` carry the
  count, and fewer than two finite refits is an `ArgumentError`. The
  excluded replicates say something about the *simulated* networks, not
  about the observed one — but they are the extreme ones. The warning, the
  note printed by `show` and `approximations(fit)` all carry the ERGM
  family's sentence: "The standard errors are conditional on a finite refit:
  the excluded replicates are the extreme ones, so the standard errors are
  biased downward." For the
  same reason the bootstrap is refused outright when the *observed* fit has
  a coefficient at `∓Inf`: the remedy is unavailable exactly where the
  uncertainty is largest.
- **Threads.** The replicates are simulated from `rng` before any refit and
  every refit is deterministic, so `threaded=false` gives bit-identical
  standard errors to the default threaded loop — the numbers never depend
  on the thread count (pinned by a test that CI runs on four threads).

```julia
using ERGMMulti, NetworkCore, Random
rng = Xoshiro(1)
m = MultilayerNetwork(20; directed = true)
add_layer!(m, :friendship); add_layer!(m, :advice)
for i in 1:20, j in 1:20
    i == j && continue
    rand(rng) < 0.15 && add_layer_edge!(m, :friendship, i, j)
    rand(rng) < 0.15 && add_layer_edge!(m, :advice, i, j)
end
fit = fit_ergm_multi(m, [LayerEdges(), LayerMutual()]; method = :mple,
                     se = :bootstrap, n_boot = 30, rng = Xoshiro(2))
se_method(fit)                       # :bootstrap
size(fit.boot_replicates)            # (30, 2)
approximations(fit)                  # the MPLE caveat and the bootstrap description
confint(fit)                         # calibrated intervals

mple = fit_ergm_multi(m, [LayerEdges(), LayerMutual()]; method = :mple)
mple.inference_withheld              # true
all(isnan, coeftable(mple).p_values) # true — no test on the naive errors
naive = fit_ergm_multi(m, [LayerEdges(), LayerMutual()]; method = :mple,
                       se = :hessian)
confint(naive)                       # R's naive Wald intervals, on request
```

## Goodness of fit

[`gof`](@ref)`(fit)` simulates `n_sim` multilayer networks at the fitted
coefficients and compares the observed network with them, panel by panel:

- `"model statistics"` and `"layer edges"`. For an MPLE fit of a
  dyad-dependent model the model statistics are *not* reproduced by
  construction (they would be at the MLE), so this panel mostly measures the
  bias of the pseudo-likelihood estimate;
- `"multiplexity"` — the number of dyads tied in exactly `k` of the `L`
  layers: the cross-layer structure;
- per layer, `"degree: <layer>"` (undirected) or `"idegree: <layer>"` and
  `"odegree: <layer>"` (directed), and `"esp: <layer>"` — the degree and
  edgewise-shared-partner distributions of `gof.ergm`, statistics the model
  does not fit directly and where misfit shows.

```julia
g = gof(fit; n_sim = 50, rng = Xoshiro(3))
[panel.name for panel in g.statistics]
```

Geodesic-distance panels are not implemented.

## Missing dyads

Every layer must be **fully observed**. The MPLE enumerates every
within-layer dyad as an observed row, so a masked (unobserved) dyad would
enter the pseudo-likelihood at its face value; `ergm_multi` therefore
calls `require_observed` on every layer and refuses, naming the layer:

> ArgumentError: ergm_multi (layer 2) does not support missing
> (unobserved) dyads, but the network has 1 masked dyad. A masked dyad is
> unobserved, not absent, so reading its face value would silently invent
> data.
>   • call `clear_missing_dyads!(net)` to declare every dyad observed; or
>   • use a routine that implements missing-data handling
>     (`NetworkCore.supports_missing(f) == true`).

There is no `missing=` keyword on the estimator
(`NetworkCore.missing_policies(ergm_multi) == (:error,)`): no policy other
than refusal is offered, because none would be honest for an estimator
that enumerates every dyad. [`simulate_multi_ergm`](@ref) refuses a masked
layer the same way (its message starts `simulate_multi_ergm (layer 2)`):
the chain would otherwise toggle the unobserved dyad at its face value.
`gof` is covered transitively — a `MultiERGMResult` never holds a masked
network. The adapters ([`combine_networks`](@ref), [`split_by_layer`](@ref),
[`as_multilayer`](@ref)) *carry* the mask rather than read it, so a masked
layer survives a round trip and is refused at the estimator, not silently
un-masked on the way.

## Defaults

The sampler's `burnin` and `interval` — on [`simulate_multi_ergm`](@ref),
on [`gof`](@ref) and as `boot_burnin`/`boot_interval` on the bootstrap —
default to `nothing`, which resolves to the ERGM family's one dyad-scaled
rule (`ERGM.Extension.mcmc_defaults`) over the **within-layer dyads**
`n_dyads = L · n · (n − 1)` (directed) or `L · n · (n − 1) / 2`
(undirected):

- `burnin = 20 × n_dyads`,
- `interval = max(100, n_dyads ÷ 10)`.

A 4-actor directed two-layer network (24 dyads) gets `burnin = 480,
interval = 100`; the 20-actor fixture (760 dyads) gets `burnin = 15200,
interval = 100`. Passing an explicit `burnin`/`interval` overrides the
rule for that call. The remaining defaults are `n_sim = 1` (sampler),
`n_sim = 100` (`gof`), `n_boot = 100`, `maxiter = 100`, `tol = 1e-8`,
`se = nothing`, `threaded = true`, and `rng = Random.default_rng()` —
every random draw flows through `rng`, so a seeded call is reproducible.

## Checking

An edges-only model reproduces `logit(density)` exactly (per layer or
pooled), and simulation→estimation round trips recover coefficients —
both are covered in the test suite, as are two provenanced golden fixtures
against real `ergm.multi` 0.3.0 output: a dyad-independent fit with and
without offsets (`test/fixtures/twolayer_ergm_multi.toml`; coefficients to
~1e-13 against R's `glm` at `epsilon = 1e-14`), and, on the same frozen
layers, `summary()` of every term with an `ergm.multi` counterpart plus the
MPLE of a dyad-dependent model with `L(~edges, ~A&B)` and `L(~mutual, ~A)`
(`test/fixtures/twolayer_layer_terms.toml`; statistics exact, coefficients
to ~1e-7 against R's own compressed design refit at `glm` epsilon 1e-14).
A third fixture (`test/fixtures/pooled_layer_terms.toml`) pins the pooled
form `L(~term, c(~A, ~B))` for thirty-four ergm terms and two MPLE fits, and
a fourth (`test/fixtures/multilayer_labels.toml`) the coefficient labels
`ergm.multi` prints for every multilayer term, offsets included. Every
fixture compares the labels exactly.
The term-by-term correspondence is tabulated in [Terms](@ref terms-guide).

Fits whose formula contains dyad-dependent terms (classified by
`ERGM.is_dyad_dependent`, extended to the multilayer terms, through the
shared `ERGM.has_dyad_dependent(model)` predicate) print the
pseudo-likelihood caveat, report `is_exact(fit) == false`, withhold the
naive Wald inference of an MPLE fit, and list the approximation in
`approximations(fit)`; `se=:bootstrap` replaces the covariance with a
parametric bootstrap.

## Maximum likelihood (MCMLE)

R's `ergm(Layer(...) ~ ...)` fits a dyad-dependent multilayer model by
Monte-Carlo maximum likelihood, and so does `ergm_multi` by default
(`method = :mcmle` asks for it explicitly):

```julia
mle = fit_ergm_multi(m, [LayerEdges(), LayerMutual()]; rng = Xoshiro(7))
mle.method, se_method(mle)           # (:mcmle, :fisher)
mle.converged
mle.mcmc.convergence                 # iterations, step length, t-ratios, Hotelling p, ESS
                                     # (of the sample drawn before the last step)
stderror(mle)                        # inverse Fisher information + Monte-Carlo error
mle.mcmc.mc_std_errors               # the Monte-Carlo part alone
loglikelihood(mle)                   # the log-likelihood, by path sampling
confint(mle)
```

Starting from the MPLE, each iteration draws `n_samples` (default 1024)
multilayer networks at the current coefficients with the sampler of
`simulate_multi_ergm` (the chain continues across iterations; offsets stay
fixed and enter every chain) and moves the free coefficients by the
Monte-Carlo Newton step `γ·Σ̂⁻¹(g(y_obs) − ḡ)`, with Hummel's step length
`γ`. The iteration is ERGM.jl's own (`ERGM.Extension.mcmle_solve`, the loop
`ERGM.mcmle` runs), so the stopping rule is R ergm 4's confidence test: at
`γ = 1`, with 99 % confidence (`conv_confidence`), the estimating equations
at the updated coefficients must lie inside the tolerance region set by
`conv_precision` (0.1); when the test fails near the solution the sample is
boosted, up to `max_n_samples`. `termination=:hotelling` selects the older
t-ratio and Hotelling T² rule at a fixed sample size. `show` prints the rule
and its p-value. The final sample — `mle.mcmc.samples`, and the t-ratios and
Hotelling p-value in `mle.mcmc.convergence` — is the one drawn at the last
iterate, *before* the final step to the returned coefficients (as in R), so
the convergence verdict is the stopping rule, not those diagnostics. A fit
that does not converge says so in the stopping rule's own terms: under the
default `termination=:confidence`, "MCMLE did not converge in k iterations
(99% equivalence test p … (needs < 0.01; tolerance precision 0.1, n draws),
step length γ …)" in the warning, `show` and `approximations`; under
`:hotelling`, the Hotelling p-value and the largest t-ratio, which are that
rule. The log-likelihood is integrated by path sampling from the
dyad-independent part of the model, whose normalizer is exact
(`bridge_rungs`, `bridge_samples`; `bridge_rungs=0` skips it), and `aic`/`bic`
are built on it. The result is reproducible from `rng` and does not depend
on the thread count.

It is validated two ways: against exact enumeration on a 3-actor two-layer
network (4096 states: the maximizer, its standard errors and the
log-likelihood), and against `ergm.multi` on a dyad-dependent two-layer
model fitted under nine R seeds (`test/fixtures/multilayer_mcmle.toml`) —
coefficients, standard errors and log-likelihood inside a band of four of R's
own seed-to-seed standard deviations. On that model the MPLE and the MLE
differ by up to 0.6 standard errors.

A dyad-independent formula under `method=:mcmle` returns the exact MPLE with
no Monte Carlo. A statistic at the boundary of its attainable range is fixed
at `∓Inf` and held there by the sampler (see *Boundary statistics and
separation*). `method=:mcmle` is refused with an `ArgumentError` when the
MPLE start is separated or unconverged (unless `init=` is given), under
`drop = false` with a boundary statistic, on a masked layer, and together
with a keyword of the MPLE (`se=`, …).

## Not implemented (known limitations)

The items — the MCMLE is not R's implementation, no models over several
networks (`Networks()`, `N()`, `gofN`), layer logic limited to `A&B` on
edges and `mutualL`, descriptive-only multilevel statistics, no geodesic GOF
panel — are stated with the exact behaviour a user gets in
[Not implemented](@ref not-implemented) on the index page (and identically
in the README and the CHANGELOG). The one that matters most here: a
boundary statistic is fixed at `∓Inf` by both estimators, where `ergm.multi`
reports `NA` or a finite value, and the MCMLE refuses a separated start,
where `ergm.multi` warns and returns a finite value.
