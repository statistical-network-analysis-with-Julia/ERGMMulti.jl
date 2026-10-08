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
network — the analogue of `ergm.multi`'s `L(~term, ~A)`.
`WithinLayer(term, [l1, l2])` (or `WithinLayer(term, :)` for every layer)
**pools** the term: the statistic is the sum of its per-layer statistics
under one coefficient, `ergm.multi`'s `L(~term, c(~A, ~B))`. `LayerEdges`,
`LayerMutual` and `LayerTriangle` are convenience forms of the same idea
(`LayerTriangle([1, 2])` is `WithinLayer(Triangle(), [1, 2])`).

```julia
using ERGMMulti, ERGM
m = MultilayerNetwork(5; directed = false)
add_layer!(m, :friendship); add_layer!(m, :advice)
for (i, j) in ((1, 2), (1, 3), (2, 3), (3, 4))
    add_layer_edge!(m, :friendship, i, j)
end
for (i, j) in ((1, 2), (2, 4), (3, 4), (2, 3))
    add_layer_edge!(m, :advice, i, j)
end
compute(WithinLayer(GWESP(0.5), 1), m)        # the friendship layer's gwesp
compute(WithinLayer(GWESP(0.5), [1, 2]), m)   # friendship + advice, one coefficient
name(WithinLayer(GWESP(0.5), [1, 2]), m)      # "L((friendship,advice))~gwesp.fixed.0.5"
name(WithinLayer(Degree(1), :), m)            # "L((friendship,advice))~degree1"
```

A pooled term is validated against, and its attributes are read from, every
selected layer. `WithinLayer(Offset(...), l)` is refused: fix a multilayer
coefficient with the `offsets=` keyword. A within-layer term that expands into several
statistics in ERGM.jl (a multi-level `NodeFactor`, a `Degree(0:2)`) expands
here too: one coefficient per statistic, `θ` of
[`simulate_multi_ergm`](@ref) has one entry per statistic, and an offset
on the whole term is refused (pin each level with its own single-statistic
term). A *pooled* expanding term must expand into the same statistics on
every selected layer (the same attribute levels); otherwise the model is
refused with an `ArgumentError` naming the layers.

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
  ("term 'L(3)~edges' refers to layer 3, but the network has 2 layers
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
  'L((:))~mutual' is only defined for directed networks, but the network is
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
- **Labels.** Coefficients carry the labels `ergm.multi` prints, the
  layers named by `layer_names(m)`. A within-layer term is `L(<layer>)~`
  followed by ERGM.jl's direction-aware label, resolved against the layer:
  on a layer named `:friendship`, `WithinLayer(GWESP(0.5), 1)` is
  `L(friendship)~gwesp.OTP.fixed.0.5` on a directed layer and
  `L(friendship)~gwesp.fixed.0.5` on an undirected one; a pool over a list
  of layers is `L((friendship,advice))~…` (R's `L(~term, c(~A, ~B))`,
  double parentheses even for one layer). The multilayer terms are
  `L(friendship)~edges`, `L(friendship)~mutual`, `L(friendship)~triangle`
  (or the pooled forms), `L(friendship&advice)~edges` and
  `L(friendship,advice)~mutual`; an offset row of `coeftable` is
  `offset(L(friendship)~edges)`. Without a network (`name(term)`, used in
  messages about a term not yet attached to data) the layers are named by
  index, `L(1)~edges`, as R names an unnamed layer list. `show`,
  `coeftable` and `gof` all use these, and
  `test/fixtures/multilayer_labels.toml` pins them against `ergm.multi`.

```julia
using ERGMMulti, ERGM
m = MultilayerNetwork(4; directed = true)
add_layer!(m, :friendship); add_layer!(m, :advice)
try
    MultiERGMModel([WithinLayer(Kstar(2), 1)], m)
catch e
    println(e.msg)      # ... only defined for undirected networks ... Use `OStar(2)` / `IStar(2)` ...
end
MultiERGMModel([WithinLayer(GWESP(0.5), 1)], m).terms.names   # ["L(friendship)~gwesp.OTP.fixed.0.5"]
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
| `LayerEdges()` / `LayerEdges([1, 2])` | `L(~edges, c(~A, ~B))` (one coefficient for `L(~edges, ~A)` + `L(~edges, ~B)`) | `multilayer_labels` (three layers: pools over all, a subset and one layer; label and statistic); `logit(density)` analytic test |
| `LayerMutual(1)` | `L(~mutual, ~A)` | `twolayer_layer_terms` (directed: statistic, MPLE) |
| `LayerTriangle(1)` | `L(~triangle, ~A)` | `twolayer_layer_terms` (directed: statistic; undirected: statistic, MPLE); `boundary_multi` (at its bound with all-zero change statistics: MPLE limit, MCMLE with the drop) |
| `WithinLayer(GWESP(0.5), 1)` | `L(~gwesp(0.5, fixed = TRUE), ~A)` (label `gwesp.OTP.fixed.0.5` on directed layers, `gwesp.fixed.0.5` on undirected) | `twolayer_layer_terms` (directed: statistic; undirected: statistic) |
| `WithinLayer(OStar(2), 2)` | `L(~ostar(2), ~B)` | `twolayer_layer_terms` (directed: statistic) |
| `WithinLayer(Kstar(2), 2)` | `L(~kstar(2), ~B)` | `twolayer_layer_terms` (undirected: statistic) |
| `WithinLayer(GWDegree(0.3), 2)` | `L(~gwdegree(0.3, fixed = TRUE), ~B)` | `twolayer_layer_terms` (undirected: statistic) |
| `WithinLayer(NodeMatch(:grp), 2)` | `L(~nodematch("grp"), ~B)` | `twolayer_layer_terms` (directed: statistic; undirected: statistic); `twolayer_ergm_multi` (per-layer `nodematch` in the exact MLE, and as `offset.coef`) |
| `WithinLayer(TwoPath(), 1)` | `L(~twopath, ~A)` | `twolayer_layer_terms` (directed: statistic) |
| `WithinLayer(term, l)`, any other ERGM.jl term | `L(~term, ~A)` | ERGM.jl's own fixtures for `term` (the layer's network is passed through unchanged) |
| `WithinLayer(term, [1, 2])` / `WithinLayer(term, :)` | `L(~term, c(~A, ~B))` (label `L((A,B))~…` on both sides) | `pooled_layer_terms` — directed: `edges`, `mutual`, `triangle`, `gwesp`, `gwdsp`, `ostar(2)`, `istar(2)`, `twopath`, `nodematch`, `nodefactor`, `nodemix`, `nodecov`, `absdiff`, `idegree(0:2)`, `odegree(1:2)`, `gwidegree`, `gwodegree`, `meandeg`, `density` (statistics; MPLE with pooled `edges`, `mutual`, `gwesp`, `nodematch`); undirected: `edges`, `triangle`, `kstar(2)`, `gwesp`, `gwdsp`, `gwnsp`, `gwdegree`, `degree(0:2)`, `concurrent`, `nodematch`, `nodefactor`, `nodemix`, `nodecov`, `absdiff`, `degrange(2, 4)` (statistics; MPLE with pooled `edges`, `gwesp`, `nodefactor`) |
| `InterlayerDependence(1, 2)` | `L(~edges, ~A&B)` (ordered dyads on directed layers, unordered on undirected) | `twolayer_layer_terms` (directed: statistic, MPLE; undirected: statistic, MPLE); `boundary_multi` (at its bound: MPLE limit, MCMLE with the drop) |
| `MultiplexMutual(1, 2)` | `mutualL(Ls = list(~A, ~B))` | `twolayer_layer_terms` (directed: statistic); `multilayer_labels` (both layer orders) |
| every term's coefficient label | `names(coef(fit))`, `names(summary(...))` | every fixture compares R's labels exactly; `multilayer_labels` covers three named layers, both co-occurrence orders, pools over a subset, over all and over one layer, and `offset(...)` |
| `offsets = Dict(k => c)` | `offset(...)` with `offset.coef = c` | `twolayer_ergm_multi` (offset fit; and the fit with the term dropped, which the offset fit must *not* reproduce); `multilayer_labels` (the `offset(L(advice)~nodematch.g)` label) |

`CrossNetEdges`, `Nestedness`, `LevelHomophily` and `CrossLevelEdge` are
descriptive statistics with no `ergm.multi` counterpart and no fixture;
they never enter a fit.
