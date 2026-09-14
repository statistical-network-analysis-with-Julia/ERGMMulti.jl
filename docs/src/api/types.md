# Types

`MultilayerNetwork{D}`, `MultiERGMModel{D}` and `MultiERGMResult{D}` carry the
directedness of the layers as a type parameter, exactly as `Network{T,D}` and
`ERGM.ERGMModel{T,D}` do; `is_directed` reads it.

```@docs
MultilayerNetwork
MultiNetwork
MultilevelNetwork
MultiERGMModel
MultiERGMResult
```

## Construction and conversion

```@docs
add_layer!
add_layer_edge!
layer_network
layer_names
as_multilayer
combine_networks
split_by_layer
add_cross_level_edge!
```
