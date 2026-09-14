# Model Estimation

[`ergm_multi`](@ref) (= [`fit_ergm_multi`](@ref), one `const` function:
the R name and the ecosystem's `fit_<model>` name) fits by maximum
pseudo-likelihood over the **within-layer dyads**: each (layer, i, j)
contributes a logistic term in `θ'Δg` with `Δg` from
[`change_stat_layer`](@ref). The pseudo-likelihood is maximized with the
shared `Networks.newton_fit` optimizer (Newton-Raphson with step-halving)
on the shared, allocation-free `Networks.logistic_derivatives` kernel —
the same code ERGM.jl's MPLE, TERGM.jl's CMPLE and ERGMRank.jl's swap MPLE
run on. Standard errors come from the inverse observed information of the
pseudo-likelihood, or from a parametric bootstrap (`se=:bootstrap`). There
is no MCMLE — see [Not implemented](@ref not-implemented).

```julia
using ERGMMulti, Networks, Random

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

result = fit_ergm_multi(m, terms)
coef(result)
stderror(result)
loglikelihood(result)      # maximized pseudo-log-likelihood
aic(result), bic(result)
nobs(result), dof(result)  # 760 within-layer dyads, 2 estimated coefficients
confint(result)            # normal-theory limits
coeftable(result)          # the R-style table `show` prints
is_exact(result)           # false — InterlayerDependence is dyad-dependent
approximations(result)     # says so, in words
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
under `InterlayerDependence`, no mutual dyad under `LayerMutual`, a
`NodeMatch` with no within-group tie in its layer — has no finite maximum
pseudo-likelihood estimate. ERGMMulti.jl applies R ergm's default
`drop=TRUE` semantics for one-mode terms. (`ergm.multi` 0.3.0 does **not**
inherit them: its layer operator does not propagate the statistic's
attainable range to `ergm.checkextreme.model`, so for the same design R
warns "The MPLE does not exist!" and returns a *finite* ≈ −17.6 with a
standard error in the thousands for the boundary term — while its finite
coefficients agree with ERGMMulti.jl's exactly. That R output is frozen as
section (iii) of `test/fixtures/twolayer_layer_terms.toml`, a pinned
divergence.) The warning is R's sentence:

> ergm_multi: observed statistic(s) duplex.1.2 are at their smallest
> attainable values. Their coefficients will be fixed at -Inf (no finite
> maximum pseudo-likelihood estimate exists; R ergm's drop=TRUE;
> ergm.multi 0.3.0 warns "The MPLE does not exist!" and returns a finite
> value with a standard error in the thousands instead — see the
> estimation guide). The remaining coefficients are estimated on the dyads
> these statistics do not touch — the exact limit of the pseudo-likelihood
> — with standard error 0 and p-value 0 recorded for the fixed ones.

("largest attainable values … fixed at +Inf" for a statistic at its
maximum.) What the result then holds:

- `coef(fit)[k] == -Inf` (or `+Inf`), `stderror(fit)[k] == 0.0`, z
  `∓Inf`, p-value `0` — the `coeftable` row is printed as such;
- the other coefficients are the MPLE on the dyads the dropped statistic
  does not touch (the exact limit of the pseudo-likelihood as the
  coefficient goes to `∓Inf`); `bic` uses that dyad count as its sample
  size and `dof` counts only the finite coefficients;
- `fit.converged` is `true` (the reduced problem converged),
  `is_exact(fit)` is `false`, and `show(fit)`/`approximations(fit)` carry
  the note "duplex.1.2 fixed at -Inf (observed statistic at its smallest
  attainable value): no finite maximum pseudo-likelihood estimate exists …";
- `se=:bootstrap` is refused on such a fit with an `ArgumentError`
  ("se=:bootstrap is not available when a coefficient is fixed at ±Inf
  …"): a network cannot be simulated at an infinite coefficient;
- `gof(fit)` and `simulate_multi_ergm(fit.model, coef(fit))` are refused
  the same way ("gof: every coefficient must be finite (got duplex.1.2 =
  -Inf): a multilayer network cannot be simulated at an infinite or NaN
  coefficient — the chain would reject every proposal and return copies of
  the observed network …"). They used to run that frozen chain silently and
  `gof` reported `p = 1` for every statistic. Remove the dropped term, as
  R's `drop=TRUE` does, refit, and assess that model.

```julia
using ERGMMulti, Networks
m = MultilayerNetwork(30; directed=true)
add_layer!(m, :friendship); add_layer!(m, :advice)
add_layer_edge!(m, :friendship, 1, 2)          # one tie, no co-occurrence
fit = ergm_multi(m, [LayerEdges(), InterlayerDependence(1, 2)])  # warns
coef(fit)[2] == -Inf                # true
stderror(fit)[2] == 0.0             # true
coef(fit)[1] ≈ log(1 / 1738)        # true: the MPLE on the 1739 untouched dyads
dof(fit)                            # 1
is_exact(fit)                       # false
```

Perfect separation by a *combination* of statistics — no single column at
its boundary, yet a linear combination of them perfectly predicts the ties —
is R's "The MPLE does not exist!". ERGM.jl's asymptote test detects it,
with or without offsets (the offset contribution is handed to the test as a
column with a fixed coefficient of 1); the warning is

> ergm_multi: the MPLE does not exist (perfect separation): the
> pseudo-likelihood has no finite maximum, and the returned coefficients
> are the point at which Newton stopped on its flat asymptote —
> arbitrarily large, with meaningless standard errors. R ergm warns "The
> MPLE does not exist!" for the same design. The fit is returned with
> `converged == false`; …

and the fit comes back with `fit.converged == false`,
`fit.separated == true`, `is_exact(fit) == false`, and the same verdict
printed by `show` (under `converged: false`) and listed by
`approximations(fit)` instead of the generic non-convergence sentence.

## Non-convergence

A Newton iteration that runs out of `maxiter` is loud: the warning

> ergm_multi: the Newton iteration did not converge within maxiter = 100
> iterations (the maximum pseudo-likelihood estimate may not exist: a
> statistic at the boundary of its attainable range, or perfect
> separation); the result reports converged = false and its point
> estimates and standard errors are unreliable

is emitted once, `fit.converged == false`, `is_exact(fit) == false`,
`show(fit)` prints the caveat right under `converged: false`, and
`approximations(fit)` lists it as its first entry. A rank-deficient design
(two copies of the same statistic, `[LayerEdges(1), LayerEdges(1)]`) is
reported the same way, plus Networks.jl's warning "newton_fit: the Hessian
at the solution is not negative definite; standard errors are undefined
(returned as NaN)" — `stderror`, `vcov` and `confint` are then `NaN`.
An unconverged fit is never returned as a fit with a footnote: the
warning, the field, the printout and the metadata all agree.

## Standard errors

- **`se=:hessian`** (default): the inverse negative pseudo-Hessian at the
  optimum. With only dyad-independent terms (`LayerEdges`,
  `WithinLayer(NodeMatch(...))`, ...) the pseudo-likelihood *is* the
  likelihood and these are the usual maximum-likelihood standard errors
  (the golden fixture holds them to ~1e-11 against R). With any
  dyad-dependent term (`LayerMutual`, `LayerTriangle`,
  `InterlayerDependence`, `MultiplexMutual`, a `WithinLayer(GWESP(...))`)
  the (layer, i, j) conditionals are multiplied as if independent, so they
  are expected to be **anticonservative**: `show` says so, and
  `approximations(fit)` lists "inverse-Hessian standard errors of the naive
  pseudo-likelihood: expected anticonservative under dependence".
- **`se=:bootstrap`**: a parametric bootstrap on the ONE shared
  `Networks.bootstrap_cov` loop. `n_boot` (default 100) multilayer networks
  are simulated from the fitted model at `θ̂` (offsets included) with
  [`simulate_multi_ergm`](@ref) — `boot_burnin`/`boot_interval` default to
  the dyad-scaled rule below, `rng` seeds the draws — each is refit with the
  same offsets, and the empirical covariance of the refitted free
  coefficients replaces `vcov`. **The point estimates are unchanged**;
  `se_method(fit)` reports `:bootstrap`; offset rows stay `NaN`.
- **Replicate exclusion.** A replicate whose refit has no finite MPLE (a
  simulated network with no triangle under `LayerTriangle`, no
  co-occurring tie under `InterlayerDependence`, a separated design) is
  excluded from the covariance and kept as a `NaN` row of
  `fit.boot_replicates` (an `n_boot × p_free` matrix; `nothing` under
  `se=:hessian`). One warning reports "k of the n_boot bootstrap refits did
  not converge … and were excluded", `show`/`approximations(fit)` carry the
  count, and fewer than two finite refits is an `ArgumentError`. The
  excluded replicates say something about the *simulated* networks, not
  about the observed one.
- **Threads.** The replicates are simulated from `rng` before any refit and
  every refit is deterministic, so `threaded=false` gives bit-identical
  standard errors to the default threaded loop — the numbers never depend
  on the thread count (pinned by a test that CI runs on four threads).

```julia
using ERGMMulti, Networks, Random
rng = Xoshiro(1)
m = MultilayerNetwork(20; directed = true)
add_layer!(m, :friendship); add_layer!(m, :advice)
for i in 1:20, j in 1:20
    i == j && continue
    rand(rng) < 0.15 && add_layer_edge!(m, :friendship, i, j)
    rand(rng) < 0.15 && add_layer_edge!(m, :advice, i, j)
end
fit = fit_ergm_multi(m, [LayerEdges(), LayerMutual()]; se = :bootstrap,
                     n_boot = 30, rng = Xoshiro(2))
se_method(fit)                       # :bootstrap
size(fit.boot_replicates)            # (30, 2)
approximations(fit)                  # the MPLE caveat and the bootstrap description
```

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
>     (`Networks.supports_missing(f) == true`).

There is no `missing=` keyword on the estimator
(`Networks.missing_policies(ergm_multi) == (:error,)`): no policy other
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
rule (`ERGM._mcmc_defaults`) over the **within-layer dyads**
`n_dyads = L · n · (n − 1)` (directed) or `L · n · (n − 1) / 2`
(undirected):

- `burnin = 20 × n_dyads`,
- `interval = max(100, n_dyads ÷ 10)`.

A 4-actor directed two-layer network (24 dyads) gets `burnin = 480,
interval = 100`; the 20-actor fixture (760 dyads) gets `burnin = 15200,
interval = 100`. Passing an explicit `burnin`/`interval` overrides the
rule for that call. The remaining defaults are `n_sim = 1` (sampler),
`n_sim = 100` (`gof`), `n_boot = 100`, `maxiter = 100`, `tol = 1e-8`,
`se = :hessian`, `threaded = true`, and `rng = Random.default_rng()` —
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
The term-by-term correspondence is tabulated in [Terms](@ref terms-guide).

Fits whose formula contains dyad-dependent terms (classified by
`ERGM.is_dyad_dependent`, extended to the multilayer terms, through the
shared `ERGM.has_dyad_dependent(model)` predicate) print the
pseudo-likelihood caveat, report `is_exact(fit) == false`, and list the
approximation in `approximations(fit)`; `se=:bootstrap` replaces the
covariance with a parametric bootstrap.

## Not implemented (known limitations)

The four items — no MCMLE, no covariate-driven multi-network models, layer
logic limited to `A&B` and `mutualL`, descriptive-only multilevel
statistics — are stated with the exact behaviour a user gets in
[Not implemented](@ref not-implemented) on the index page (and identically
in the README and the CHANGELOG). The one that matters most here: **there
is no `method=:mcmle`**; a dyad-dependent multilayer fit is an MPLE point
estimate with the caveat above, `is_exact(fit) == false`, and
`se=:bootstrap` as the remedy for the standard errors.
