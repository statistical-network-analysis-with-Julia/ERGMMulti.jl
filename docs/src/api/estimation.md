# Estimation and Simulation

`fit_ergm_multi` is the ecosystem's harmonised `fit_<model>` name and
`ergm_multi` the R `ergm.multi` name; they are one `const` function. The
pre-0.2 `fit_multi_ergm` is a deprecated wrapper.

```@docs
fit_ergm_multi
ergm_multi
fit_multi_ergm
simulate_multi_ergm
```

## StatsAPI surface

Every verb of the ecosystem's StatsAPI surface is implemented on
[`MultiERGMResult`](@ref): `coef`, `stderror`, `vcov`, `confint`,
`loglikelihood`, `nobs`, `dof`, `aic`, `bic` and `coeftable` (pinned by
`Networks.check_statsapi(fit; strict=true)` in the test suite).

```@docs
coef(::MultiERGMResult)
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
standard error `0`, `dof` counts only finite non-offset coefficients, and
`loglikelihood` is a *pseudo*-log-likelihood unless `is_exact(fit)`.

## Goodness of Fit

ERGMMulti extends the ecosystem's single `gof` generic (defined in Networks.jl)
with a method for a multi-network fit.

```@docs
gof(::MultiERGMResult)
```
