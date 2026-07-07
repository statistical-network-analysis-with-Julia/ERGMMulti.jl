# Terms

All multilayer terms implement `compute(term, m)` and
[`change_stat_layer`](@ref)`(term, m, l, i, j)` — the **add-direction**
change in the statistic when edge (i, j) is added in layer `l`,
independent of that dyad's current state (the multilayer analogue of
ERGM.jl's change-statistic convention). The tests verify every change
statistic against brute-force recomputation.

## Layer selectors

`LayerEdges`, `LayerMutual`, and `LayerTriangle` accept an `Int` (one
layer), a `Vector{Int}`, or `:` (all layers) — vectors and `:` pool the
coefficient across layers.

## Within-layer terms

[`WithinLayer`](@ref)`(term, l)` lifts any ERGM.jl term into layer `l`,
reusing the base term's validated `compute`/`change_stat` on that layer's
network. `LayerEdges(l)`, `LayerMutual(l)`, `LayerTriangle(l)` are
convenience forms of the same idea with pooling support.

## Cross-layer terms

- [`InterlayerDependence`](@ref)`(l1, l2)`: the number of dyads with an
  edge in both layers (entrainment). Positive coefficients mean a tie in
  one layer predicts the same tie in the other.
- [`MultiplexMutual`](@ref)`(l1, l2)`: ordered dyads with `i→j` in `l1`
  and `j→i` in `l2` (cross-layer exchange/reciprocity). Directed only.
