# Types API Reference

This page documents the core data types in ERGMMulti.jl.

## Multi-Network Structures

### MultiNetwork

```@docs
MultiNetwork
```

### MultilayerNetwork

```@docs
MultilayerNetwork
```

### MultilevelNetwork

```@docs
MultilevelNetwork
```

## Structure Queries

### Vertex Count

```@docs
Graphs.nv(::MultiNetwork)
Graphs.nv(::MultilayerNetwork)
```

### Edge Operations

```@docs
total_edges
add_layer!
add_layer_edge!
has_layer_edge
```

### Layer Queries

```@docs
n_layers
layer_names
```

### Level Queries

```@docs
n_levels
```

## Model Types

### MultiERGMModel

```@docs
MultiERGMModel
```

### MultiERGMResult

```@docs
MultiERGMResult
```
