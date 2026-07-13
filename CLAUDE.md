# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ERGMMulti.jl is a Julia package for fitting Exponential Random Graph Models (ERGMs) to multiple, multilayer, and multilevel network data. It is a port of the R `ergm.multi` package from the StatNet collection.

## Development Commands

- **Run tests**: `julia --project -e 'using Pkg; Pkg.test()'`
- **Load package locally**: `julia --project -e 'using ERGMMulti'`
- **Build docs**: `julia --project=docs docs/make.jl`
- **Dev install**: `using Pkg; Pkg.develop(path=".")`

## Architecture

The entire package lives in a single source file: `src/ERGMMulti.jl`. It is organized into these sections:

1. **Data Structures** — `MultilayerNetwork` (same actors, L layers; all layers share size and directedness), `MultiNetwork` (independent networks, descriptive), `MultilevelNetwork` (per-level networks + membership Dicts keyed by each level's own node IDs + explicit cross-level edges via `add_cross_level_edge!`).
2. **Multilayer ERGM Terms** — `LayerEdges/LayerMutual/LayerTriangle` take a layer selector (Int, Vector{Int}, or Colon for pooled); `WithinLayer(term, l)` lifts any ERGM.jl term (reusing its validated change_stat); `InterlayerDependence(l1,l2)` (co-occurrence) and `MultiplexMutual(l1,l2)` (cross-layer reciprocity, ordered dyads). Every term implements `compute(term, m)` and `change_stat_layer(term, m, l, i, j)` — the ADD-DIRECTION change for adding (i,j) in layer l, state-independent (multilayer analogue of ERGM.jl's convention).
3. **Multilevel ERGM Terms** — `Nestedness`, `CrossLevelEdge`, `LevelHomophily`.
4. **Model & Estimation** — `ergm_multi()` (alias `fit_multi_ergm`): real MPLE (logistic pseudo-likelihood) over the WITHIN-LAYER dyad universe with `offsets::Dict{Int,Float64}` support (fixed per-term coefficients, ergm.multi's offset mechanism); maximized with the shared `ERGM.newton_fit` optimizer (Newton-Raphson with step-halving). Edges-only fits reproduce logit(density) exactly. `ERGM.is_dyad_dependent` is extended to the multilayer terms (`LayerEdges` false, `WithinLayer` delegates; the conservative `true` fallback covers the rest — cross-layer terms count as dyad-dependent because the model's dyads are (layer, i, j) triples), and `show(::MultiERGMResult)` prints a pseudo-likelihood caveat for dyad-dependent formulas.

   **Standard errors** (`se_type` on `MultiERGMResult`, reported by `Networks.se_method`): `se=:hessian` (default) is the inverse negative pseudo-Hessian — anticonservative for any dyad-dependent term over the `(layer, i, j)` dyad universe. `se=:bootstrap` is a **parametric bootstrap** (simulate `n_boot` multilayer networks at the fitted coefficients *including the offsets* with `simulate_multi_ergm`, refit with the same offsets via the shared `_multi_mple_fit` core, empirical covariance of the FREE coefficients), with the same API as `ERGM.mple`'s (`n_boot`/`boot_burnin`/`boot_interval`/`rng`) and running on the ONE shared `Networks.bootstrap_cov` loop — never paste the loop in. The point estimates are identical under both; only the covariance changes, and offset rows stay `NaN` under either (a fixed coefficient carries no uncertainty). `show`/`approximations` read `se_type` and drop the anticonservatism caveat when a bootstrap was actually used, while keeping the *point-estimate* MPLE caveat, which holds either way.
5. **Utilities** — `as_multilayer`; `combine_networks` builds the true BLOCK-DIAGONAL combined Network (n·L vertices, `:layer`/`:actor` vertex attributes, edges within blocks); `split_by_layer` inverts it (rejects cross-block edges).
6. **Simulation** — `simulate_multi_ergm`: Metropolis sampler restricted to within-layer dyads; addition accepted with min(1, exp(θ'Δg)), removal with the negation.

All term types subtype `AbstractERGMTerm` and extend the ERGM generics via `import ERGM: name, compute, change_stat`.

## The MPLE derivative loop is ERGM.jl's, not ours (review finding 15)

`_multi_mple_fit` used to carry its own logistic loop with `hess .-= (pr*(1-pr)) .* (x * x')` inside it: a fresh `pf×pf` outer product on every one of the `n_dyads` rows of every Newton evaluation — **649 KB and 0.80 ms per evaluation** on a 2-layer, 40-actor design. The pseudo-likelihood over the within-layer dyads **is** a logistic likelihood with an offset, so the derivatives now come from the shared **`ERGM.logistic_derivatives(Xf, y; offset=η0)`** (the same builder TERGM's CMPLE and ERGMRank's swap MPLE run on): workspaces allocated once, `η = Xfβ` by gemv and `−H = Xf'WXf` by gemm, **304 bytes and 0.10 ms per evaluation** — 7.8x faster, and the allocations no longer scale with the dyad count. `_multi_mple_design(m, terms, offsets, free) -> (Xf, y, η0)` is the design builder, factored out so the `@allocated` regression test can measure the closure the *fitter* builds. **Never paste the loop back in**; if it needs a feature, add it to `ERGM.logistic_derivatives`. The summation order moves from row-wise accumulation to BLAS, so the arithmetic is not bit-identical — but the *fit* is: measured against the old loop on the same design, **max|Δθ| = 1.1e-16**, one ulp. (Newton's last step is quadratically convergent; a last-ulp difference in the gradient and Hessian does not move its fixed point.) The golden fixture agrees at 5.0e-13 in the coefficients and 1.4e-11 in the SEs — unmoved, against a 1e-6 exact-MLE tolerance.

## Golden fixtures (statnet ergm.multi)

`test/fixtures/twolayer_ergm_multi.toml`, generated by `test/fixtures/r/twolayer_ergm_multi.R` and loaded with Networks.jl's `load_golden` (which throws without `[provenance]`). R model: `Networks(list(net1, net2)) ~ N(~edges + nodematch("grp"), ~factor(.NetworkID) - 1)` — per-layer edges and nodematch — which is exactly `[LayerEdges(1), LayerEdges(2), WithinLayer(NodeMatch(:grp), 1), WithinLayer(NodeMatch(:grp), 2)]` here. Both layers are frozen as edge lists.

Every term is dyad-independent, so the likelihood factorizes over the `(layer, i, j)` dyads and both packages compute the same **exact MLE**. ERGMMulti.jl reproduces a tightly-converged (`epsilon = 1e-14`) R `glm` to **~1e-13** in the coefficients and **~1e-11** in the standard errors. (ergm's own MPLE `glm` stops at the default `epsilon = 1e-8` and is 1.3e-9 / 4.8e-6 from the optimum — measured, and the as-shipped tolerances are set from the measurement.)

**Offsets are validated against the bug that actually happens.** A second frozen fit pins the nodematch coefficients at `offset.coef = (0.75, 0.40)`; ERGMMulti.jl's `offsets=Dict(3=>0.75, 4=>0.40)` reproduces it to ~1e-13. The fixture *also* freezes the fit of the same model with **no nodematch term at all** — where a fit lands if the offsets are dropped from the *linear predictor* rather than merely from the parameter vector — and the testset asserts the offset fit is nowhere near it (−1.418 vs −1.030). A test that only checked "matches R" could pass while both sides silently did nothing.

## Key Dependencies

- **ERGM.jl** — Base ERGM framework; provides `AbstractERGMTerm`
- **Networks.jl** — Network data structure (`Network{T}`, edge/vertex operations)
- **Graphs.jl** — Graph primitives (`nv`, `ne`, `edges`, `has_edge`, etc.)
- **LinearAlgebra**, **Statistics** — numerics for the Newton solver

## Conventions

- Julia 1.12+ required (see `Project.toml` compat).
- Types are parameterized by vertex type `T` (typically `Int`).
- Layer and network names use `Symbol` (e.g., `:friendship`, `:advice`).
- Each ERGM term struct implements `name(t)::String` and `compute(t, data)::Float64`.
- Multilayer change statistics use `change_stat_layer(t, m, layer, i, j)` (add-direction, state-independent); brute-force verified in tests.
- Functions that mutate use the `!` suffix convention (`add_layer!`, `add_layer_edge!`).
- Exports are declared at the top of the module; no re-exports from dependencies.
