# Multilevel Networks

[`MultilevelNetwork`](@ref) represents hierarchical designs: one network
per level, membership maps (`membership[l][node] = higher-level unit`,
keyed by each level's own node IDs), and explicit cross-level ties added
with [`add_cross_level_edge!`](@ref).

The associated statistics are **descriptive** (they do not enter
`ergm_multi`):

- [`Nestedness`](@ref)`(level)`: the proportion of level-`level` edges
  whose endpoints share a higher-level unit;
- [`LevelHomophily`](@ref)`(level)`: the count of such edges;
- [`CrossLevelEdge`](@ref): the number of recorded cross-level ties.

```julia
people = network(5; directed = false)
orgs = network(2; directed = false)
membership = [Dict(1 => 1, 2 => 1, 3 => 2, 4 => 2, 5 => 2)]
ml = MultilevelNetwork([people, orgs], membership)

compute(Nestedness(1), ml)
add_cross_level_edge!(ml, 1, 3, 2, 1)
compute(CrossLevelEdge(), ml)
```
