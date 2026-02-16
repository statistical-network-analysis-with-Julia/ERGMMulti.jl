# Estimation

ERGMMulti.jl estimates multi-network ERGM parameters using maximum pseudo-likelihood estimation (MPLE). This guide covers the estimation process, configuration options, and interpretation of results.

## Overview

The estimation process follows these steps:

1. **Compute observed statistics** - Calculate each term's statistic on the observed data
2. **Optimize** - Iteratively update parameter estimates to match observed statistics
3. **Compute standard errors** - Estimate the precision of coefficient estimates
4. **Return results** - Package coefficients, SEs, and diagnostics

## Fitting Models

### Basic Usage

The primary function is `ergm_multi`:

```julia
result = ergm_multi(data, terms)
```

Where `data` can be a `MultiNetwork`, `MultilayerNetwork`, or `MultilevelNetwork`, and `terms` is a vector of ERGM terms.

### Fitting Options

```julia
result = ergm_multi(data, terms;
    method = :mple,      # Estimation method
    maxiter = 100         # Maximum optimization iterations
)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | Estimation method (currently `:mple`) | `:mple` |
| `maxiter` | Maximum number of iterations | `100` |

## Maximum Pseudo-Likelihood Estimation

MPLE treats each potential edge as an independent observation and models its probability conditional on all other edges. This is computationally fast but may underestimate standard errors compared to MCMLE.

### How MPLE Works

For each possible dyad $(i,j)$ in the network:

1. Compute the change statistic $\Delta g(y)_{ij}$ for each term
2. Model the edge probability as: $\text{logit}(P(Y_{ij} = 1 | Y_{-ij})) = \boldsymbol{\theta}^\top \Delta \mathbf{g}(\mathbf{y})_{ij}$
3. Estimate $\boldsymbol{\theta}$ via logistic regression

### Advantages and Limitations

| Aspect | MPLE |
|--------|------|
| **Speed** | Very fast, even for large networks |
| **Consistency** | Consistent for dyad-independent models |
| **Standard errors** | May be too narrow for complex models |
| **Implementation** | Straightforward optimization |
| **Best for** | Exploratory analysis, sparse networks |

## Understanding Results

The `MultiERGMResult` object contains:

| Field | Type | Description |
|-------|------|-------------|
| `model` | `MultiERGMModel` | The model specification |
| `coefficients` | `Vector{Float64}` | Estimated coefficients |
| `std_errors` | `Vector{Float64}` | Standard errors |
| `loglik` | `Float64` | Log-likelihood at convergence |
| `converged` | `Bool` | Whether optimization converged |

### Displaying Results

```julia
println(result)

# Output:
# Multi-Network ERGM Results
# ==========================
# Log-likelihood: -234.5678
# Converged: true
#
# Coefficients:
#   edges.friendship               -1.2345 (SE: 0.1234)
#   edges.advice                   -1.5678 (SE: 0.1456)
#   mutual.friendship               0.8901 (SE: 0.2345)
#   interlayer.friendship.advice     0.4567 (SE: 0.1789)
```

### Accessing Results

```julia
# Coefficient vector
result.coefficients

# Standard errors
result.std_errors

# Log-likelihood
result.loglik

# Check convergence
result.converged

# Access model specification
result.model.terms     # The terms used
result.model.data      # The data object
```

## Interpreting Coefficients

### Log-Odds Ratios

Coefficients are log-odds ratios, the same as in standard ERGMs. A coefficient $\theta_k$ for term $k$ means:

- **$\theta_k > 0$**: The term increases the probability of an edge
- **$\theta_k < 0$**: The term decreases the probability of an edge
- **$\exp(\theta_k)$**: The odds ratio associated with a one-unit increase in the statistic

### Example Interpretations

| Term | Coefficient | Interpretation |
|------|-------------|----------------|
| `edges.friendship = -2.0` | Friendship ties are sparse (low baseline density) |
| `mutual.friendship = 1.5` | Strong reciprocity in friendship (odds ratio 4.5) |
| `interlayer.friendship.advice = 0.8` | Friends are 2.2x more likely to also have advice ties |
| `multiplex.mutual = 0.5` | Cross-layer reciprocity: advice flows back to friends |
| `nested.1 = 1.2` | Ties are 3.3x more likely within teams |

### Confidence Intervals

Approximate 95% confidence intervals:

```julia
using Distributions

z = quantile(Normal(), 0.975)

lower = result.coefficients .- z .* result.std_errors
upper = result.coefficients .+ z .* result.std_errors

for (i, term) in enumerate(result.model.terms)
    println("$(name(term)): [$(round(lower[i], digits=3)), $(round(upper[i], digits=3))]")
end
```

## Model Comparison

### Comparing Log-Likelihoods

```julia
# Model 1: Within-layer only
terms1 = [LayerEdges(:friendship), LayerEdges(:advice)]

# Model 2: Add cross-layer effects
terms2 = [LayerEdges(:friendship), LayerEdges(:advice),
          InterlayerDependence(:friendship, :advice)]

result1 = ergm_multi(mln, terms1)
result2 = ergm_multi(mln, terms2)

println("Model 1 LL: ", result1.loglik)
println("Model 2 LL: ", result2.loglik)
```

### Incremental Model Building

A good strategy is to build the model incrementally:

```julia
# Step 1: Density
terms_step1 = [LayerEdges(:friendship), LayerEdges(:advice)]

# Step 2: Add within-layer structure
terms_step2 = [terms_step1..., LayerMutual(:friendship), LayerMutual(:advice)]

# Step 3: Add cross-layer effects
terms_step3 = [terms_step2..., InterlayerDependence(:friendship, :advice)]

# Fit each and compare
for (i, terms) in enumerate([terms_step1, terms_step2, terms_step3])
    r = ergm_multi(mln, terms)
    println("Step $i: LL=$(round(r.loglik, digits=2)), converged=$(r.converged)")
end
```

## Convergence Issues

### Checking Convergence

```julia
if !result.converged
    @warn "Model did not converge - results may be unreliable"
end
```

### Common Causes and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Sparse layers | Non-convergence | Simplify model, remove sparse layer terms |
| Perfect prediction | Very large coefficients | Remove or constrain problematic terms |
| Too many terms | Slow convergence | Reduce model complexity |
| Collinear terms | Large standard errors | Remove correlated terms |

### Handling Non-Convergence

```julia
# Increase iterations
result = ergm_multi(mln, terms; maxiter=500)

# Check for problematic coefficients
for (i, term) in enumerate(result.model.terms)
    if abs(result.coefficients[i]) > 10
        @warn "Possible issue with $(name(term)): coef = $(result.coefficients[i])"
    end
end
```

## Simulation

### Simulating from Fitted Models

```julia
# Simulate networks from fitted model (placeholder)
simulated = simulate_multi_ergm(result; n_sim=10)
```

Note: Full simulation support for multi-network ERGMs is still under development. Currently returns copies of the observed data.

## Best Practices

1. **Check convergence**: Always verify `result.converged == true`
2. **Build incrementally**: Start simple and add terms one at a time
3. **Include density terms**: Always include `LayerEdges` for each layer
4. **Watch for separation**: Very large coefficients suggest model misspecification
5. **Compare models**: Use log-likelihoods to evaluate model improvement
6. **Start with MPLE**: Use pseudo-likelihood for initial exploration
7. **Examine all layers**: Ensure each layer has enough edges for estimation
8. **Report uncertainty**: Always report standard errors alongside coefficients
