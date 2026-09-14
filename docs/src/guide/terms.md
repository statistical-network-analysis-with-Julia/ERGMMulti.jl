# [Terms](@id terms-guide)

All multilayer terms implement `compute(term, m)` and
[`change_stat_layer`](@ref)`(term, m, l, i, j)` — the **add-direction**
change in the statistic when edge (i, j) is added in layer `l`,
independent of that dyad's current state (the multilayer analogue of
ERGM.jl's change-statistic convention). The tests verify every change
statistic against brute-force recomputation on a directed and an
undirected fixture.

```julia
using ERGMMulti, ERGM
m = MultilayerNetwork(4; directed = true)
add_layer!(m, :friendship); add_layer!(m, :advice)
add_layer_edge!(m, :friendship, 1, 2); add_layer_edge!(m, :friendship, 2, 1)
add_layer_edge!(m, :advice, 1, 2);     add_layer_edge!(m, :advice, 2, 3)
compute(LayerEdges(1), m)                                  # 2.0
compute(LayerMutual(1), m)                                 # 1.0
compute(InterlayerDependence(1, 2), m)                     # 1.0 — 1→2 is in both
change_stat_layer(InterlayerDependence(1, 2), m, 1, 2, 3)  # 1.0 — 2→3 exists in :advice
```

## Layer selectors

`LayerEdges`, `LayerMutual`, and `LayerTriangle` accept an `Int` (one
layer), a `Vector{Int}`, or `:` (all layers) — vectors and `:` pool the
coefficient across layers. A pooled term is the statistic of the
block-diagonal combined network (the sum over the selected layers under
one coefficient).

## Within-layer terms

[`WithinLayer`](@ref)`(term, l)` lifts any ERGM.jl term into layer `l`,
reusing the base term's validated `compute`/`change_stat` on that layer's
network — the analogue of `ergm.multi`'s `L(~term, ~A)`. `LayerEdges(l)`,
`LayerMutual(l)`, `LayerTriangle(l)` are convenience forms of the same idea
with pooling support. A within-layer term that expands into several
statistics in ERGM.jl (a multi-level `NodeFactor`, a `Degree(0:2)`) expands
here too: one coefficient per statistic, `θ` of
[`simulate_multi_ergm`](@ref) has one entry per statistic, and an offset
on the whole term is refused (pin each level with its own single-statistic
term).

## Cross-layer terms

- [`InterlayerDependence`](@ref)`(l1, l2)`: the number of dyads with an
  edge in both layers (entrainment). Positive coefficients mean a tie in
  one layer predicts the same tie in the other.
- [`MultiplexMutual`](@ref)`(l1, l2)`: ordered dyads with `i→j` in `l1`
  and `j→i` in `l2` (cross-layer exchange/reciprocity). Directed only — on
  an undirected `MultilayerNetwork{false}` it is refused with ERGM.jl's
  directed-only error rather than silently contributing `0`.

Both count **ordered** dyads on directed layers, as `ergm.multi` does (the
fixture below pins that on a case where ordered and unordered counting
differ). These two are the only cross-layer forms implemented; see
[Not implemented](@ref not-implemented) for the rest of `ergm.multi`'s
layer logic.

## Validation

Every term is checked against the data when the [`MultiERGMModel`](@ref)
is built — which both [`ergm_multi`](@ref) and
[`simulate_multi_ergm`](@ref) do first — and a problem is an
`ArgumentError` that names the fix, before any fitting or sampling:

- **Layer indices.** Every layer a term refers to (`LayerEdges(3)`,
  `InterlayerDependence(1, 3)`, `WithinLayer(t, 3)`) must exist; the
  message quotes `layer_names(m)` so the index can be corrected
  ("term 'L.edges.3' refers to layer 3, but the network has 2 layers
  (:friendship, :advice); layer indices must lie in 1:2").
- **What `WithinLayer(t, l)` checks against layer `l`** — ERGM.jl's own
  formula validator, so the errors are the ones ERGM.jl and R raise: a
  vertex attribute the layer lacks (`WithinLayer(NodeMatch(:welth), 1)`
  → "refers to vertex attribute :welth, which does not exist on the
  network"), an attribute set on only some vertices (statnet refuses NA:
  "needs vertex attribute :grp on every vertex, but 1 of 4 vertices have
  no value"), an `EdgeCov` whose matrix is not `n × n`, and the direction
  requirement below. Attribute terms are then materialized into dense
  snapshots, so a later change to the attribute is not seen by a model
  already built.
- **Directed-only and undirected-only terms.** As in R ergm and
  `ergm.multi`, `Kstar`, `GWDegree` and `Degree` are undirected-only and
  `Mutual`, `OStar`, `IStar`, `GWODegree`, `GWIDegree` directed-only
  (`GWESP`/`GWDSP` work on both, with the OTP label on directed layers); the multilayer `LayerMutual` and
  `MultiplexMutual` are directed-only. On a layer of the wrong kind the
  model constructor, `compute` and `change_stat_layer` all throw ERGM.jl's
  sentence ("term 'kstar2' is only defined for undirected networks, but
  the network is directed. Use `OStar(2)` / `IStar(2)` instead"; "term
  'L.mutual.all' is only defined for directed networks, but the network is
  undirected") — never a silent `0.0`.
- **A bare ERGM.jl term.** `Mutual()` where `WithinLayer(Mutual(), l)` was
  meant — the ergm.multi migrant's `L(~mutual, ~A)` slip — has no
  multilayer change statistic and is refused by the model constructor
  ("term 'mutual' (Mutual) has no multilayer change statistic
  (`change_stat_layer`): it is a single-network ERGM.jl term or a multilevel
  descriptive, not a multilayer term. In a multilayer model lift an ERGM.jl
  term into a layer with `WithinLayer(Mutual(...), l)` (the analogue of
  ergm.multi's `L(~mutual, ~A)`), or use the pooled/per-layer forms …"). It
  used to construct and fail at fit time with a `MethodError`. The
  multilevel descriptives (`Nestedness`, …) are refused by the same check.
- **Layers are selected by index.** `LayerEdges(:advice)`,
  `WithinLayer(Triangle(), :advice)`, `InterlayerDependence(:friendship,
  :advice)` and `MultiplexMutual(:friendship, 2)` throw an `ArgumentError`
  at the term's construction with the index to use ("LayerEdges selects
  layers by index, not by name (got :advice): use LayerEdges(k) with
  k = findfirst(==(:advice), layer_names(m)) …") — the migration note the
  CHANGELOG records, at the point where it is needed. Names are accepted by
  `add_layer_edge!` and `layer_network`; terms are built before they meet a
  network, so they carry indices.
- **Self-loops and two-mode layers.** A layer that contains a self-loop is
  refused by [`MultiERGMModel`](@ref) (a `loops=true` layer with none is
  accepted), and a two-mode (bipartite) layer by `add_layer!`; both
  sentences are in [Model Estimation](@ref) under *Validation before
  fitting*.
- **Labels.** The coefficient of a within-layer term is `L<l>.` followed
  by ERGM.jl's direction-aware label, resolved against the layer:
  `WithinLayer(GWESP(0.5), 1)` is `L1.gwesp.OTP.fixed.0.5` on a directed
  layer and `L1.gwesp.fixed.0.5` on an undirected one, exactly as ERGM.jl
  and R label them; the multilayer terms are `L.edges.<sel>`,
  `L.mutual.<sel>`, `L.triangle.<sel>`, `duplex.<l1>.<l2>` and
  `duplex.mutual.<l1>.<l2>`. `show`, `coeftable` and `gof` all use these.

```julia
using ERGMMulti, ERGM
m = MultilayerNetwork(4; directed = true)
add_layer!(m, :friendship); add_layer!(m, :advice)
try
    MultiERGMModel([WithinLayer(Kstar(2), 1)], m)
catch e
    println(e.msg)      # ... only defined for undirected networks ... Use `OStar(2)` / `IStar(2)` ...
end
MultiERGMModel([WithinLayer(GWESP(0.5), 1)], m).terms.names   # ["L1.gwesp.OTP.fixed.0.5"]
```

## Correspondence with `ergm.multi`

Each row names the R term a statistic reproduces and the golden fixture
that pins it (both are generated by checked-in scripts under
`test/fixtures/r/` from real `ergm.multi` 0.3.0 output, with a
`[provenance]` block; the layers are `Layer(list(A = net1, B = net2))`).
"Statistic" means `summary()` agrees exactly (tolerance 1e-10); "MPLE"
means the term also enters a fit whose coefficients and standard errors
are held to R's own design refit (1e-6 / 1e-4, 20x the measured `glm`
slack). **Directedness matters**: `twolayer_ergm_multi` and sections
(i)–(iii) of `twolayer_layer_terms` use two *directed* 20-actor layers
("directed" below); section (iv) of `twolayer_layer_terms` pins the
*undirected* branch — unordered within-layer dyads (`nobs = L·n(n−1)/2`,
`ergmMPLE`'s weights summing to 132), unordered co-occurrence for
`L(~edges, ~A&B)`, and the undirected `triangle`/`gwesp`/`kstar`/`gwdegree`
forms — on two undirected 12-actor layers drawn from the same seed
("undirected" below). A term with no "undirected" entry is directed-only
(`LayerMutual`, `MultiplexMutual`, `OStar`, `TwoPath`) or is pinned on
directed layers only.

| ERGMMulti.jl | `ergm.multi` | Pinned by |
|:---|:---|:---|
| `LayerEdges(1)` | `L(~edges, ~A)` | `twolayer_layer_terms` (directed: statistic, MPLE; undirected: statistic, MPLE); `twolayer_ergm_multi` (`N(~edges, ~factor(.NetworkID) - 1)`, exact MLE, with and without offsets) |
| `LayerEdges()` / `LayerEdges([1, 2])` | `edges` on the combined `Layer()` network (one coefficient for `L(~edges, ~A)` + `L(~edges, ~B)`) | the sum of the per-layer statistics above; `logit(density)` analytic test |
| `LayerMutual(1)` | `L(~mutual, ~A)` | `twolayer_layer_terms` (directed: statistic, MPLE) |
| `LayerTriangle(1)` | `L(~triangle, ~A)` | `twolayer_layer_terms` (directed: statistic; undirected: statistic, MPLE) |
| `WithinLayer(GWESP(0.5), 1)` | `L(~gwesp(0.5, fixed = TRUE), ~A)` (label `gwesp.OTP.fixed.0.5` on directed layers, `gwesp.fixed.0.5` on undirected) | `twolayer_layer_terms` (directed: statistic; undirected: statistic) |
| `WithinLayer(OStar(2), 2)` | `L(~ostar(2), ~B)` | `twolayer_layer_terms` (directed: statistic) |
| `WithinLayer(Kstar(2), 2)` | `L(~kstar(2), ~B)` | `twolayer_layer_terms` (undirected: statistic) |
| `WithinLayer(GWDegree(0.3), 2)` | `L(~gwdegree(0.3, fixed = TRUE), ~B)` | `twolayer_layer_terms` (undirected: statistic) |
| `WithinLayer(NodeMatch(:grp), 2)` | `L(~nodematch("grp"), ~B)` | `twolayer_layer_terms` (directed: statistic; undirected: statistic); `twolayer_ergm_multi` (per-layer `nodematch` in the exact MLE, and as `offset.coef`) |
| `WithinLayer(TwoPath(), 1)` | `L(~twopath, ~A)` | `twolayer_layer_terms` (directed: statistic) |
| `WithinLayer(term, l)`, any other ERGM.jl term | `L(~term, ~A)` | ERGM.jl's own fixtures for `term` (the layer's network is passed through unchanged) |
| `InterlayerDependence(1, 2)` | `L(~edges, ~A&B)` (ordered dyads on directed layers, unordered on undirected) | `twolayer_layer_terms` (directed: statistic, MPLE; undirected: statistic, MPLE) |
| `MultiplexMutual(1, 2)` | `mutualL(Ls = list(~A, ~B))` | `twolayer_layer_terms` (directed: statistic) |
| `offsets = Dict(k => c)` | `offset(...)` with `offset.coef = c` | `twolayer_ergm_multi` (offset fit; and the fit with the term dropped, which the offset fit must *not* reproduce) |

`CrossNetEdges`, `Nestedness`, `LevelHomophily` and `CrossLevelEdge` are
descriptive statistics with no `ergm.multi` counterpart and no fixture;
they never enter a fit.
