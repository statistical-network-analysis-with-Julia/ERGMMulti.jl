# Terms

Terms in ERGMMulti.jl capture structural effects within and across network layers. All terms subtype `AbstractERGMTerm` and implement a common interface, so they can be freely combined in models.

## Term Interface

All terms implement two methods:

```julia
compute(term, data) -> Float64
name(term) -> String
```

The `compute` function calculates the full statistic value for the term given the multi-network data. The `name` function returns a human-readable identifier.

## Term Categories

ERGMMulti.jl organizes terms into five categories:

| Type | Description | Examples |
|------|-------------|----------|
| **Within-Layer** | Effects within a single layer | LayerEdges, LayerMutual, LayerTriangle |
| **Wrapper** | Apply standard ERGM terms to layers | WithinLayer |
| **Between-Layer** | Cross-layer dependence effects | InterlayerDependence, MultiplexMutual |
| **Multi-Network** | Effects across independent networks | CrossNetEdges |
| **Multilevel** | Hierarchical nesting effects | Nestedness, CrossLevelEdge, LevelHomophily |

## Within-Layer Terms

These capture structural effects within a single layer of a multilayer network.

### LayerEdges

Edge count within a specific layer:

```julia
# Number of edges in the friendship layer
LayerEdges(:friendship)
```

**Interpretation**: A negative coefficient indicates lower density than expected by chance (the typical case). Controls for baseline density in each layer.

### LayerMutual

Mutuality (reciprocated edges) within a specific layer:

```julia
# Reciprocated ties in the friendship layer
LayerMutual(:friendship)
```

**Interpretation**: A positive coefficient indicates that ties tend to be reciprocated within the layer - if $i \to j$, then $j \to i$ is more likely.

### LayerTriangle

Triangle count within a specific layer:

```julia
# Transitive triangles in the friendship layer
LayerTriangle(:friendship)
```

Visual representation:

```text
  k
 ↗ ↘
i → j   ← all three edges within the same layer
```

**Interpretation**: A positive coefficient indicates triadic closure - friends of friends tend to become friends within the layer.

## WithinLayer Wrapper

The `WithinLayer` wrapper applies any standard ERGM term from ERGM.jl to a specific layer. This is the most flexible way to specify within-layer effects.

```julia
using ERGM

# Apply standard terms within layers
WithinLayer(Edges(), :friendship)       # Same as LayerEdges(:friendship)
WithinLayer(GWESP(0.5), :friendship)    # GWESP within friendship layer
WithinLayer(GWDegree(0.5), :advice)     # Degree distribution in advice layer
```

**Use case**: When you need ERGM terms beyond the built-in LayerEdges, LayerMutual, and LayerTriangle.

### Example: Rich Within-Layer Model

```julia
terms = [
    # Friendship layer
    WithinLayer(Edges(), :friendship),
    WithinLayer(GWESP(0.5), :friendship),
    WithinLayer(GWDegree(0.5), :friendship),

    # Advice layer
    WithinLayer(Edges(), :advice),
    WithinLayer(GWESP(0.5), :advice),
]
```

## Between-Layer Terms

These capture statistical dependence between edges in different layers - the core of multiplex network analysis.

### InterlayerDependence

Tendency for edges in one layer given edges in the other:

```julia
# Friendship edges predict advice edges
InterlayerDependence(:friendship, :advice)
```

**Formula**: Counts dyads $(i,j)$ with edges in both layers:

$$g_{\text{interlayer}}(\mathbf{y}) = \sum_{i \neq j} y_{ij}^{(\text{friendship})} \cdot y_{ij}^{(\text{advice})}$$

**Interpretation**: A positive coefficient indicates that having a tie in one layer increases the probability of a tie in the other layer for the same dyad.

### MultiplexMutual

Cross-layer reciprocity - edge $(i,j)$ in layer 1 and edge $(j,i)$ in layer 2:

```julia
# If i→j in friendship, then j→i in advice
MultiplexMutual(:friendship, :advice)
```

Visual representation:

```text
Layer 1 (friendship):   i → j
Layer 2 (advice):       j → i
```

**Interpretation**: A positive coefficient indicates cross-layer reciprocity. For example, people reciprocate friendship ties with advice-seeking (or vice versa).

### BetweenLayers

Count of edges that span layers (in networks with interlayer connections):

```julia
BetweenLayers(:friendship, :advice)
```

**Use case**: When your multilayer network has explicit connections between layers (not just parallel edges on the same dyads).

## Multi-Network Terms

For `MultiNetwork` data with multiple independent networks.

### CrossNetEdges

Total edges across all networks in a MultiNetwork:

```julia
CrossNetEdges()
```

**Interpretation**: Controls for overall edge density across pooled networks. Analogous to `Edges()` in a standard ERGM.

### Example: Multi-Network Model

```julia
# Multiple classroom networks
mn = MultiNetwork([class1, class2, class3]; names=[:A, :B, :C])

terms = [
    CrossNetEdges(),     # Shared density parameter
]

result = ergm_multi(mn, terms)
```

## Multilevel Terms

For `MultilevelNetwork` data with hierarchical structure.

### Nestedness

Tendency for edges to exist within groups at a given level:

```julia
# Within-team ties at level 1
Nestedness(1)
```

Visual representation:

```text
Team 1:  [A ─ B ─ C]     ← edges within team
Team 2:  [D ─ E]          ← edges within team
         A ─── D          ← edge across teams (not counted)
```

**Interpretation**: A positive coefficient indicates that actors within the same group are more likely to be tied.

### CrossLevelEdge

Edges that span levels in the hierarchy:

```julia
# Edges between level 1 (individuals) and level 2 (teams)
CrossLevelEdge(1, 2)
```

**Interpretation**: Captures the tendency for cross-level connections.

### LevelHomophily

Homophily based on group membership at a given level:

```julia
# Same-team homophily
LevelHomophily(1)
```

**Interpretation**: Tests whether actors in the same group at a given level are more likely to form ties at the lowest level. Similar to `Nestedness` but always applied to the lowest-level network.

## Using Terms in Practice

### Building a Multilayer Model

```julia
using Network
using ERGMMulti

# Create multilayer network
mln = MultilayerNetwork{Int}(50; layer_names=[:friendship, :advice])

# Populate layers
for (i, j) in [(1,2), (2,3), (3,4), (4,5)]
    add_layer_edge!(mln, :friendship, i, j)
end
for (i, j) in [(1,3), (2,4), (3,5)]
    add_layer_edge!(mln, :advice, i, j)
end

# Build model with within-layer and between-layer effects
terms = [
    # Within-layer effects
    LayerEdges(:friendship),
    LayerEdges(:advice),
    LayerMutual(:friendship),
    LayerTriangle(:friendship),

    # Between-layer effects
    InterlayerDependence(:friendship, :advice),
    MultiplexMutual(:friendship, :advice),
]

# Fit
result = ergm_multi(mln, terms)
```

### Incremental Model Building

A recommended approach is to build models incrementally:

```julia
# Model 1: Within-layer density only
terms1 = [
    LayerEdges(:friendship),
    LayerEdges(:advice),
]

# Model 2: Add within-layer structural effects
terms2 = [
    LayerEdges(:friendship),
    LayerEdges(:advice),
    LayerMutual(:friendship),
    LayerMutual(:advice),
]

# Model 3: Add between-layer effects
terms3 = [
    LayerEdges(:friendship),
    LayerEdges(:advice),
    LayerMutual(:friendship),
    LayerMutual(:advice),
    InterlayerDependence(:friendship, :advice),
    MultiplexMutual(:friendship, :advice),
]

# Fit and compare
result1 = ergm_multi(mln, terms1)
result2 = ergm_multi(mln, terms2)
result3 = ergm_multi(mln, terms3)

println("Model 1 LL: ", result1.loglik)
println("Model 2 LL: ", result2.loglik)
println("Model 3 LL: ", result3.loglik)
```

## Choosing Terms

### By Research Question

| Question | Terms |
|----------|-------|
| What is the density of each layer? | LayerEdges |
| Is there reciprocity within layers? | LayerMutual |
| Is there triadic closure within layers? | LayerTriangle, WithinLayer(GWESP(...)) |
| Do ties in one layer predict ties in another? | InterlayerDependence |
| Is there cross-layer reciprocity? | MultiplexMutual |
| Do actors cluster within groups? | Nestedness, LevelHomophily |
| Are there cross-level ties? | CrossLevelEdge |

### Best Practices

1. **Start with density**: Always include LayerEdges for each layer
2. **Add reciprocity**: LayerMutual is almost always relevant for directed layers
3. **Test cross-layer effects carefully**: InterlayerDependence can be correlated with within-layer density
4. **Use WithinLayer for advanced effects**: GWESP and GWDegree capture richer within-layer structure
5. **Include relevant multilevel terms**: Based on domain knowledge about hierarchical structure
6. **Avoid multicollinearity**: Don't include highly correlated terms (e.g., Nestedness and LevelHomophily at the same level)
