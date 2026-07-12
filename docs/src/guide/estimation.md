# Model Estimation

[`ergm_multi`](@ref) fits by maximum pseudo-likelihood over the
**within-layer dyads**: each (layer, i, j) contributes a logistic term in
`θ'Δg` with `Δg` from [`change_stat_layer`](@ref). The pseudo-likelihood
is maximized with the shared `ERGM.newton_fit` optimizer (Newton-Raphson
with step-halving); standard errors come from the inverse observed
information of the pseudo-likelihood.

```julia
using ERGMMulti, Network, Random

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

result = ergm_multi(m, terms)
result.coefficients
result.std_errors
result.loglik      # maximized pseudo-log-likelihood
result.aic, result.bic
```

## Offsets

`offsets = Dict(k => c)` fixes term `k`'s coefficient at `c` — the
`ergm.multi` offset mechanism used for per-layer size adjustments or
theory-fixed effects. Offset terms report their fixed coefficient with
`NaN` standard error; the free coefficients are estimated with the offset
contribution absorbed into the linear predictor.

## Checking

An edges-only model reproduces `logit(density)` exactly (per layer or
pooled), and simulation→estimation round trips recover coefficients —
both are covered in the test suite. As with any pseudo-likelihood,
standard errors understate uncertainty under strong dependence —
fitted results whose formula contains dyad-dependent terms (classified
by `ERGM.is_dyad_dependent`, extended to the multilayer terms) print an
explicit caveat.
