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

Statistics are labelled as R `ergm.multi` labels them, resolved against the
network with ERGM.jl's two-argument `name(term, net)` generic, so a directed
layer's `WithinLayer(GWESP(0.5), 1)` prints as `L1.gwesp.OTP.fixed.0.5` and an
undirected one as `L1.gwesp.fixed.0.5`.

```@docs
ERGMMulti.name(::WithinLayer, ::MultilayerNetwork)
```
