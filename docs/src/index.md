# ERGMMulti.jl

ERGMs for multilayer networks: several relations on the same actors,
modeled jointly following R `ergm.multi` (Krivitsky, Koehly & Marcum
2020).

## The block-diagonal mechanism

The layers of a [`MultilayerNetwork`](@ref) combine into one
block-diagonal network ([`combine_networks`](@ref)) whose vertices carry
`:layer` and `:actor` attributes. The model's dyad universe is the
within-layer dyads; layer-aware terms then express per-layer effects,
pooled effects, and cross-layer dependence.

## Contents

```@contents
Pages = [
    "getting_started.md",
    "guide/structures.md",
    "guide/terms.md",
    "guide/estimation.md",
    "guide/multilevel.md",
    "api/types.md",
    "api/terms.md",
    "api/estimation.md",
]
Depth = 2
```

## References

1. Krivitsky, P.N., Koehly, L.M. & Marcum, C.S. (2020). Exponential-family
   random graph models for multi-layer networks. *Psychometrika*, 85(3), 630-659.
