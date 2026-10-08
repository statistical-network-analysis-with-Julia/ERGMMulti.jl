# Estimation and Simulation

`fit_ergm_multi` is the ecosystem's harmonised `fit_<model>` name and
`ergm_multi` the R `ergm.multi` name; they are one `const` function.

```@docs
fit_ergm_multi
ergm_multi
simulate_multi_ergm
```

## StatsAPI surface

Every verb of the ecosystem's StatsAPI surface is implemented on
[`MultiERGMResult`](@ref): `coef`, `coefnames`, `stderror`, `vcov`,
`confint`, `loglikelihood`, `nobs`, `dof`, `aic`, `bic` and `coeftable`
(pinned by `NetworkCore.check_statsapi(fit; required=(STATSAPI_VERBS...,
:coefnames), strict=true)` in the test suite).

```@docs
coef(::MultiERGMResult)
coefnames(::MultiERGMResult)
stderror(::MultiERGMResult)
vcov(::MultiERGMResult)
confint(::MultiERGMResult)
coeftable(::MultiERGMResult)
loglikelihood(::MultiERGMResult)
nobs(::MultiERGMResult)
dof(::MultiERGMResult)
aic(::MultiERGMResult)
bic(::MultiERGMResult)
```

Every accessor reads the fit as it was actually made: an offset term's
standard error is `NaN`, a boundary statistic's coefficient is `∓Inf` with
standard error `0`, a statistic with no identifiable coefficient is `NaN`,
`dof` counts only finite non-offset coefficients, and `loglikelihood` is a
*pseudo*-log-likelihood for an MPLE fit unless `is_exact(fit)`.

## Goodness of Fit

ERGMMulti extends the ecosystem's single `gof` generic (defined in NetworkCore.jl)
with a method for a multi-network fit.

```@docs
gof(::MultiERGMResult)
```

## Renamed and removed names

0.2.0 is the first release. These names of the development versions were
removed without a deprecation period; there is no alias for them.

| Removed | Use instead |
|:--|:--|
| `fit_multi_ergm(m, terms; ...)` | `fit_ergm_multi(m, terms; ...)` or `ergm_multi(m, terms; ...)` |
| coefficient labels `L.edges.1`, `L.edges.all`, `L.mutual.1`, `L.triangle.1` | `ergm.multi`'s labels with the layer names: `L(friendship)~edges`, `L((friendship,advice))~edges`, `L(friendship)~mutual`, `L(friendship)~triangle` |
| `L1.<term>`, `L1+2.<term>`, `Lall.<term>` (`WithinLayer`) | `L(friendship)~<term>`, `L((friendship,advice))~<term>` |
| `duplex.1.2` (`InterlayerDependence`) | `L(friendship&advice)~edges` |
| `duplex.mutual.1.2` (`MultiplexMutual`) | `L(friendship,advice)~mutual` |
| `<label> (offset)` (offset rows of `coeftable`) | `offset(<label>)` |
| the MPLE as the default for every formula | `method = :auto` (R's rule); pass `method = :mple` for the MPLE of a dyad-dependent formula |
