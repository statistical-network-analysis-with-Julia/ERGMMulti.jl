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

1. **Data Structures** — `MultiNetwork` (collection of independent networks), `MultilayerNetwork` (same nodes, multiple edge types/layers), `MultilevelNetwork` (hierarchical nesting with membership vectors).
2. **Multilayer ERGM Terms** — `LayerEdges`, `LayerMutual`, `LayerTriangle` (within-layer); `InterlayerDependence`, `MultiplexMutual`, `BetweenLayers` (between-layer); `WithinLayer` (wrapper to apply any standard ERGM term to a layer); `CrossNetEdges` (multi-network).
3. **Multilevel ERGM Terms** — `Nestedness`, `CrossLevelEdge`, `LevelHomophily`.
4. **Model & Estimation** — `MultiERGMModel`, `MultiERGMResult`, `ergm_multi()` entry point with MPLE estimation via `multi_mple()`. `fit_multi_ergm` is an alias for `ergm_multi`.
5. **Utilities** — `as_multilayer`, `combine_networks`, `split_by_layer`.
6. **Simulation** — `simulate_multi_ergm` (placeholder, not fully implemented).

All term types are subtypes of `AbstractERGMTerm` (from the ERGM package) and implement `name()` and `compute()` methods.

## Key Dependencies

- **ERGM.jl** — Base ERGM framework; provides `AbstractERGMTerm`
- **Network.jl** — Network data structure (`Network{T}`, edge/vertex operations)
- **Graphs.jl** — Graph primitives (`nv`, `ne`, `edges`, `has_edge`, etc.)
- **Optim.jl**, **LinearAlgebra**, **Statistics**, **StatsBase** — Numerical optimization and statistics

## Conventions

- Julia 1.9+ required (see `Project.toml` compat).
- Types are parameterized by vertex type `T` (typically `Int`).
- Layer and network names use `Symbol` (e.g., `:friendship`, `:advice`).
- Each ERGM term struct implements `name(t)::String` and `compute(t, data)::Float64`.
- Change statistics use `change_stat(t, data, layer, i, j)` where applicable.
- Functions that mutate use the `!` suffix convention (`add_layer!`, `add_layer_edge!`).
- Exports are declared at the top of the module; no re-exports from dependencies.
