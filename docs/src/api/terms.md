# Terms

```@docs
LayerEdges
LayerMutual
LayerTriangle
WithinLayer
InterlayerDependence
MultiplexMutual
CrossNetEdges
Nestedness
CrossLevelEdge
LevelHomophily
change_stat_layer
```

## Coefficient labels

Coefficients carry the labels `ergm.multi` prints, with the layers named by
`layer_names(m)`. A within-layer statistic is labelled `L(<layer>)~`
followed by R's label for the term, resolved against the layer's network
with ERGM.jl's two-argument `name(term, net)` generic: on layers named `:A`
and `:B`, a directed layer's `WithinLayer(GWESP(0.5), 1)` prints as
`L(A)~gwesp.OTP.fixed.0.5`, an undirected one as `L(A)~gwesp.fixed.0.5`, and
the pooled `WithinLayer(GWESP(0.5), [1, 2])` as `L((A,B))~gwesp.fixed.0.5`.
The multilayer terms are `L(A)~edges`, `L((A,B))~edges` (pooled),
`L(A)~mutual`, `L(A)~triangle`, `L(A&B)~edges` (`InterlayerDependence`) and
`L(A,B)~mutual` (`MultiplexMutual`); an offset row is `offset(L(A)~edges)`.
Without a network (`name(term)`) the layers are named by index — `L(1)~edges`,
which is also what R prints for an unnamed layer list. The labels are pinned
against `ergm.multi` by `test/fixtures/multilayer_labels.toml`.

```@docs
ERGMMulti.name(::WithinLayer, ::MultilayerNetwork)
```
