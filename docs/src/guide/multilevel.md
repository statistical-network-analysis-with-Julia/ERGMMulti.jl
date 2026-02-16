# Multilevel Networks

Multilevel networks represent hierarchical structures where actors at one level are nested within groups at higher levels. ERGMMulti.jl provides the `MultilevelNetwork` type and associated ERGM terms for modeling these hierarchical dependencies.

## Overview

In many organizational and social settings, networks exist at multiple levels simultaneously:

```text
Level 3: Department1 ─── Department2 ─── Department3
              │                │               │
Level 2: Team1 Team2    Team3 Team4     Team5 Team6
           │     │        │     │         │     │
Level 1: Individuals ── connected ── by ── ties
```

ERGMMulti.jl models these structures by representing networks at each level and tracking group membership across levels.

## Creating Multilevel Networks

### Two-Level Structure

The simplest case: individuals nested in groups.

```julia
using Network
using ERGMMulti

# Level 1: Individual ties (20 people)
individual = Network{Int}(; n=20, directed=true)
add_edge!(individual, 1, 2)
add_edge!(individual, 2, 3)
add_edge!(individual, 6, 7)
add_edge!(individual, 7, 8)
add_edge!(individual, 11, 12)

# Level 2: Group ties (4 groups)
group = Network{Int}(; n=4, directed=false)
add_edge!(group, 1, 2)
add_edge!(group, 2, 3)

# Membership: 5 people per group
membership = [1, 1, 1, 1, 1,  # People 1-5 in Group 1
              2, 2, 2, 2, 2,  # People 6-10 in Group 2
              3, 3, 3, 3, 3,  # People 11-15 in Group 3
              4, 4, 4, 4, 4]  # People 16-20 in Group 4

# Create multilevel network
mln = MultilevelNetwork([individual, group], [membership])
println("Levels: ", n_levels(mln))  # 2
```

### Three-Level Structure

For deeper hierarchies, such as employees in teams in departments:

```julia
# Level 1: Individual ties (60 people)
individual = Network{Int}(; n=60, directed=true)
# ... add edges ...

# Level 2: Team ties (12 teams)
team = Network{Int}(; n=12, directed=false)
# ... add edges ...

# Level 3: Department ties (3 departments)
department = Network{Int}(; n=3, directed=false)
add_edge!(department, 1, 2)

# Membership mappings
ind_to_team = repeat(1:12, inner=5)    # 5 people per team
team_to_dept = repeat(1:3, inner=4)     # 4 teams per department

mln = MultilevelNetwork(
    [individual, team, department],
    [ind_to_team, team_to_dept]
)

println("Levels: ", n_levels(mln))  # 3
```

### Validation Rules

The `MultilevelNetwork` constructor enforces:

- `length(levels) == length(membership) + 1` - One membership vector per level transition
- Each membership vector maps actors at level $l$ to groups at level $l+1$

## Accessing Multilevel Structure

### Querying Levels

```julia
# Number of levels
n_levels(mln)

# Access network at each level
mln.levels[1]   # Individual network
mln.levels[2]   # Group/team network
mln.levels[3]   # Department network (if 3 levels)

# Vertex counts at each level
nv(mln.levels[1])  # Number of individuals
nv(mln.levels[2])  # Number of groups

# Edge counts at each level
ne(mln.levels[1])  # Individual-level ties
ne(mln.levels[2])  # Group-level ties
```

### Querying Membership

```julia
# Which group does individual i belong to?
group_id = mln.membership[1][i]

# Which department does team t belong to?
dept_id = mln.membership[2][t]

# All individuals in group g
group_members = findall(==(g), mln.membership[1])
```

## Multilevel ERGM Terms

### Nestedness

The `Nestedness` term captures the tendency for edges to exist within groups at a specified level:

```julia
# Within-group ties at level 1
Nestedness(1)

# Within-department ties at level 2 (for 3-level structure)
Nestedness(2)
```

**How it works**: Counts edges $(i,j)$ where both $i$ and $j$ share the same group membership at the specified level.

**Interpretation**: A positive coefficient indicates that actors in the same group are more likely to form ties, beyond what the baseline density would predict.

### LevelHomophily

Similar to `Nestedness`, but always examines edges at the lowest (individual) level:

```julia
# Same-group homophily
LevelHomophily(1)

# Same-department homophily (for 3-level structure)
LevelHomophily(2)
```

**Difference from Nestedness**: `Nestedness(l)` counts edges within level $l$'s network that stay within groups. `LevelHomophily(l)` always looks at individual-level edges and counts those where both endpoints share group membership at level $l$.

### CrossLevelEdge

Counts edges that span hierarchical levels:

```julia
# Individual-to-group boundary crossing
CrossLevelEdge(1, 2)

# Group-to-department boundary crossing
CrossLevelEdge(2, 3)
```

**Use case**: Tests whether actors form ties that cross organizational boundaries. A negative coefficient would suggest that cross-level ties are rare compared to within-level ties.

## Example: Organizational Analysis

```julia
using Network
using ERGMMulti

# Setup: 40 employees in 8 teams in 2 divisions
n_people = 40
n_teams = 8
n_divisions = 2

# Individual communication network
comm = Network{Int}(; n=n_people, directed=true)

# Within-team communication (high density)
using Random
Random.seed!(42)
for team in 1:n_teams
    members = ((team-1)*5+1):(team*5)
    for i in members, j in members
        i != j && rand() < 0.4 && add_edge!(comm, i, j)
    end
end

# Cross-team communication (low density)
for i in 1:n_people, j in 1:n_people
    i != j && !has_edge(comm, i, j) && rand() < 0.02 && add_edge!(comm, i, j)
end

# Team collaboration network
teams = Network{Int}(; n=n_teams, directed=false)
add_edge!(teams, 1, 2)
add_edge!(teams, 3, 4)
add_edge!(teams, 5, 6)

# Division network
divisions = Network{Int}(; n=n_divisions, directed=false)
add_edge!(divisions, 1, 2)

# Membership
person_to_team = repeat(1:n_teams, inner=5)
team_to_division = repeat(1:n_divisions, inner=4)

mln = MultilevelNetwork(
    [comm, teams, divisions],
    [person_to_team, team_to_division]
)

# Multilevel model
terms = [
    Nestedness(1),            # Within-team ties
    Nestedness(2),            # Within-division ties
    LevelHomophily(1),        # Team-based homophily
    CrossLevelEdge(1, 2),     # Team boundary crossing
]

result = ergm_multi(mln, terms)
println(result)

# Expected: Nestedness(1) strongly positive (within-team ties common)
# Expected: CrossLevelEdge negative (cross-team ties rare)
```

## Design Considerations

### Choosing the Right Level Structure

| Scenario | Levels | Example |
|----------|--------|---------|
| People in groups | 2 | Students in classrooms |
| People in groups in organizations | 3 | Employees in teams in departments |
| Nested communities | 2+ | Villages in districts in regions |

### Membership Encoding

Membership vectors should use contiguous integer IDs starting from 1:

```julia
# Correct: contiguous IDs
membership = [1, 1, 1, 2, 2, 2, 3, 3, 3]

# Incorrect: gaps in IDs
membership = [1, 1, 1, 5, 5, 5, 10, 10, 10]  # Will cause issues
```

### Combining with Multilayer Terms

You can model both multilayer and multilevel structures by using terms from both categories:

```julia
# If your data has both layers and levels
terms = [
    LayerEdges(:friendship),          # Layer density
    InterlayerDependence(:friendship, :advice),  # Cross-layer
    Nestedness(1),                     # Within-group
    LevelHomophily(1),                # Group homophily
]
```

## Best Practices

1. **Start with nestedness**: `Nestedness(1)` is usually the most important multilevel effect
2. **Add levels incrementally**: Test one level at a time before combining
3. **Check group sizes**: Groups should have enough members for meaningful estimation
4. **Avoid redundancy**: `Nestedness` and `LevelHomophily` at the same level may be collinear
5. **Consider directionality**: Individual-level ties may be directed even if group-level ties are not
6. **Validate membership**: Ensure every individual is assigned to exactly one group
