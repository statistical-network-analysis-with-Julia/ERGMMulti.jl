# Terms API Reference

This page documents all ERGM terms available in ERGMMulti.jl.

## Within-Layer Terms

Terms that operate within a single layer of a multilayer network.

```@docs
LayerEdges
LayerMutual
LayerTriangle
WithinLayer
```

## Between-Layer Terms

Terms that capture dependencies between different layers.

```@docs
InterlayerDependence
MultiplexMutual
BetweenLayers
```

## Multi-Network Terms

Terms for collections of independent networks.

```@docs
CrossNetEdges
```

## Multilevel Terms

Terms for hierarchical network structures.

```@docs
Nestedness
CrossLevelEdge
LevelHomophily
```

## Term Interface

All terms implement the standard ERGM term interface:

```@docs
name
compute
change_stat
```
