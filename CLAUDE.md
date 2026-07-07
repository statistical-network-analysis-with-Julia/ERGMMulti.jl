# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ERGMMulti.jl is a Julia package for fitting Exponential Random Graph Models (ERGMs) to multiple, multilayer, and multilevel network data. It is a port of the R `ergm.multi` package from the StatNet collection.

## Development Commands

- **Run tests**: `julia --project -e 'using Pkg; Pkg.test()'`
- **Load package locally**: `julia --project -e 'using ERGMMulti'`
- **Build docs**: `julia --project=docs docs/make.jl`
- **Dev install**: `using Pkg; Pkg.develop(path=".")`

## Architecture

The entire package lives in a single source file: `src/ERGMMulti.jl`. It is organized into these sections:

1. **Data Structures** — `MultilayerNetwork` (same actors, L layers; all layers share size and directedness), `MultiNetwork` (independent networks, descriptive), `MultilevelNetwork` (per-level networks + membership Dicts keyed by each level's own node IDs + explicit cross-level edges via `add_cross_level_edge!`).
2. **Multilayer ERGM Terms** — `LayerEdges/LayerMutual/LayerTriangle` take a layer selector (Int, Vector{Int}, or Colon for pooled); `WithinLayer(term, l)` lifts any ERGM.jl term (reusing its validated change_stat); `InterlayerDependence(l1,l2)` (co-occurrence) and `MultiplexMutual(l1,l2)` (cross-layer reciprocity, ordered dyads). Every term implements `compute(term, m)` and `change_stat_layer(term, m, l, i, j)` — the ADD-DIRECTION change for adding (i,j) in layer l, state-independent (multilayer analogue of ERGM.jl's convention).
3. **Multilevel ERGM Terms** — `Nestedness`, `CrossLevelEdge`, `LevelHomophily`.
4. **Model & Estimation** — `ergm_multi()` (alias `fit_multi_ergm`): real MPLE (logistic pseudo-likelihood) over the WITHIN-LAYER dyad universe with `offsets::Dict{Int,Float64}` support (fixed per-term coefficients, ergm.multi's offset mechanism); Newton-Raphson with step-halving. Edges-only fits reproduce logit(density) exactly.
5. **Utilities** — `as_multilayer`; `combine_networks` builds the true BLOCK-DIAGONAL combined Network (n·L vertices, `:layer`/`:actor` vertex attributes, edges within blocks); `split_by_layer` inverts it (rejects cross-block edges).
6. **Simulation** — `simulate_multi_ergm`: Metropolis sampler restricted to within-layer dyads; addition accepted with min(1, exp(θ'Δg)), removal with the negation.

All term types subtype `AbstractERGMTerm` and extend the ERGM generics via `import ERGM: name, compute, change_stat`.

## Key Dependencies

- **ERGM.jl** — Base ERGM framework; provides `AbstractERGMTerm`
- **Network.jl** — Network data structure (`Network{T}`, edge/vertex operations)
- **Graphs.jl** — Graph primitives (`nv`, `ne`, `edges`, `has_edge`, etc.)
- **LinearAlgebra**, **Statistics** — numerics for the Newton solver

## Conventions

- Julia 1.12+ required (see `Project.toml` compat).
- Types are parameterized by vertex type `T` (typically `Int`).
- Layer and network names use `Symbol` (e.g., `:friendship`, `:advice`).
- Each ERGM term struct implements `name(t)::String` and `compute(t, data)::Float64`.
- Multilayer change statistics use `change_stat_layer(t, m, layer, i, j)` (add-direction, state-independent); brute-force verified in tests.
- Functions that mutate use the `!` suffix convention (`add_layer!`, `add_layer_edge!`).
- Exports are declared at the top of the module; no re-exports from dependencies.
