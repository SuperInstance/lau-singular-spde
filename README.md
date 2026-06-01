# lau-singular-spde

**Singular stochastic PDE formulation of agent learning dynamics with Hairer's regularity structures and renormalization.**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-126-green.svg)]()

## What This Does

This crate formulates agent learning dynamics as **singular stochastic PDEs** and applies Martin Hairer's **regularity structures** framework to make sense of them. When many agents learn in continuous time with noise, their collective belief evolution takes the form:

```
∂ₜu = Lu + F(u, ∇u, ...) + σξ
```

where `L` is a linear operator (diffusion), `F` is a nonlinearity, and `ξ` is space-time white noise. The problem: in spatial dimensions ≥ 2, the products `u²`, `u³`, `(∇u)²` are **ill-defined** because `u` itself is only a distribution, not a function. This crate provides the renormalization machinery (Wick ordering, regularity structures, renormalization group) to give these products rigorous meaning.

**126 tests** cover every module from white noise generation through Wick ordering, RG fixed points, and universal class classification.

## Key Idea

Different agent learning architectures — gradient descent, natural gradient, mirror descent, policy gradient, actor-critic, meta-learning, attention, Bayesian updating, evolutionary strategies — fall into **universality classes** just like physical systems near critical points. A neural network doing policy gradient and a population of evolutionary strategies can exhibit *the same large-scale learning dynamics* if they share the same fixed point of the renormalization group flow.

The three universal classes are:

| Class | SPDE | Agent Behavior | Singularity |
|-------|------|----------------|-------------|
| **KPZ** | `∂ₜh = ν∇²h + λ(∇h)² + ξ` | Competitive growth dynamics | Singular in d ≥ 1 |
| **Allen-Cahn** | `∂ₜφ = Δφ + φ - φ³ + ξ` | Bistable phase transitions | Singular in d ≥ 2 |
| **Φ⁴** | `∂ₜφ = Δφ - m²φ - λ:φ³: + ξ` | Critical learning | Singular in d ≥ 1 |

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-singular-spde = { git = "https://github.com/SuperInstance/lau-singular-spde" }
```

### Dependencies

- `nalgebra` — linear algebra (matrices, vectors)
- `serde` / `serde_json` — serialization
- `rand` / `rand_distr` — random number generation for noise
- `thiserror` — error types

## Quick Start

```rust
use lau_singular_spde::*;

// Define an agent SPDE: the KPZ equation in 1 spatial dimension
let spde = AgentSPDE {
    dimension: 64,           // number of grid points (= number of agents)
    spatial_dim: 1,           // 1D spatial domain
    diffusion: nalgebra::DMatrix::identity(64, 64),  // ν·I
    nonlinearity: NonlinearityType::GradientSquared,  // (∇u)²
    noise_intensity: 0.5,
    mass: 0.0,
    coupling: 0.3,
};

// Classify its singularity
let class = classify_singularity(&spde);
assert_eq!(class, SingularityClass::Singular);

// Compute the theoretical Hölder regularity
let regularity = holder_exponent(1, NonlinearityType::GradientSquared);
// KPZ in 1d: α ≈ 0.0 (barely a function!)

// Check if renormalization is needed
assert!(needs_renormalization(&regularity));

// Apply Wick ordering to renormalize divergent products
let wick = WickProduct::new(2);  // :u²:
let renormalized = wick.evaluate(1.5, 0.5);  // u² - ⟨u²⟩ = 2.25 - 0.5 = 1.75

// Classify a PLATO agent type
let classification = classify_agent(PlatoAgentType::PolicyGradient, 64);
println!("Policy gradient → {} (needs renormalization: {})",
    classification.universal_class,
    classification.needs_renormalization);
```

## API Reference

### `spde` — Stochastic PDE Formulation

| Type | Description |
|------|-------------|
| `AgentSPDE` | An SPDE representing agent learning: dimension, spatial dimension, diffusion matrix, nonlinearity type, noise intensity, mass, coupling |
| `WhiteNoise` | Space-time white noise realization on a grid with configurable intensity |
| `NonlinearityType` | `Linear`, `Quadratic`, `Cubic`, `GradientSquared`, `Polynomial(n)` |
| `SingularityClass` | `Regular`, `Critical`, or `Singular` |
| `classify_singularity()` | Determine singularity class from SPDE parameters |
| `solve_spde()` | Euler-Maruyama time-stepping solver |
| `energy_functional()` | Compute the energy functional E[u] = ∫ (½|∇u|² + V(u)) dx |

### `regularity` — Hairer's Regularity Structures

| Type | Description |
|------|-------------|
| `Symbol` | An abstract symbol in the regularity structure (e.g., `Ξ` for noise, `I(Ξ)` for convolved noise) with a regularity exponent |
| `ModelledDistribution` | An element of the model distribution space T_α: coefficients indexed by symbols |
| `Model` | Concrete realization: maps abstract symbols to concrete distributions on a grid |
| `StructureGroup` | The group G acting on T (represented as linear transformations) |
| `ReconstructionOperator` | Maps modelled distributions back to honest distributions |

Key symbols and their regularity exponents:
- `Ξ` (noise): regularity = -(d/2 + 1)
- `I(Ξ)` (heat kernel convolution): regularity = -(d/2 + 1) + 2
- Products `τ₁·τ₂`: regularity = α₁ + α₂
- Integration `I(τ)`: regularity = α + 2

### `wick` — Wick Ordering & Renormalization

| Type | Description |
|------|-------------|
| `WickProduct` | A Wick-ordered product `:uⁿ:` with subtraction coefficients computed via Hermite polynomials |
| `WickProduct::new(n)` | Create `:uⁿ:` — subtractions follow Hermite polynomial recursion Hₙ |
| `wick.evaluate(u, σ²)` | Compute `:uⁿ:` at point `u` with variance `σ²` |
| `wick.evaluate_vector(values, σ²)` | Apply Wick ordering to a full grid of values |

Renormalization formulas:
```
:u²: = u² - σ²
:u³: = u³ - 3σ²u
:u⁴: = u⁴ - 6σ²u² + 3σ⁴
```

### `rg` — Renormalization Group Flow

| Type | Description |
|------|-------------|
| `CouplingSpace` | A point in the space of coupling constants (mass, coupling λ, noise σ²) |
| `BetaFunction` | β(λ) = dλ/d(ln scale): `Phi4SuperRenorm`, `WilsonFisher`, `KPZ`, `Custom` |
| `FixedPoint` | A fixed point of the RG flow with stability classification (`Gaussian`, `WilsonFisher`, `Trivial`) |
| `RGFlow` | Compute the full RG trajectory by integrating β-functions |
| `FixedPointStability` | `Relevant`, `Irrelevant`, `Marginal` |

The Wilson-Fisher fixed point at `λ* = ε/b` (where ε = 4 - d) governs critical learning behavior.

### `universal` — Universal Agent Learning Classes

| Type | Description |
|------|-------------|
| `UniversalClass` | `KPZ`, `AllenCahn`, `Phi4`, `Diffusive`, `MeanField` |
| `KPZDynamics` | KPZ equation with viscosity, coupling, noise; includes Tracy-Widom fluctuation statistics |
| `AllenCahnDynamics` | Allen-Cahn with double-well potential; phase transition dynamics |
| `Phi4Dynamics` | Φ⁴ with Wick-ordered cubic term; critical learning |
| `classify_universal()` | Classify an SPDE into its universality class |

### `holders` — Hölder Regularity of Belief Trajectories

| Function | Description |
|----------|-------------|
| `holder_exponent(dim, nonlinearity)` | Theoretical Hölder exponent for SPDE solution |
| `empirical_holder_exponent(trajectory)` | Estimate α from simulated trajectory data |
| `needs_renormalization(regularity)` | Whether the solution requires Wick ordering |

Regularity guide:
- Linear SHE in d=1: α = +½ (barely a function!)
- Linear SHE in d=2: α = -1 (distribution)
- KPZ in d=1: α ≈ 0 (on the boundary)
- Φ⁴ in d=1: α ≈ -½ (definitely a distribution)

### `plato` — PLATO Agent Classification

| Type | Description |
|------|-------------|
| `PlatoAgentType` | 10 agent types: `GradientDescent`, `NaturalGradient`, `MirrorDescent`, `PolicyGradient`, `ActorCritic`, `MetaLearner`, `Ensemble`, `Attention`, `Bayesian`, `Evolutionary` |
| `PlatoClassification` | Full classification: singularity class, universal class, Hölder regularity, Wick order needed |
| `classify_agent()` | Map any PLATO agent type to its SPDE classification |

## How It Works

### 1. Formulate Agent Dynamics as an SPDE

Agent beliefs `u(x,t)` evolve on a spatial grid (agents at positions `x`). The update rule determines the nonlinearity:
- **Gradient descent** → linear diffusion (`F = 0`)
- **Natural gradient** → quadratic correction (`F(u) = λu²`)
- **Mirror descent** → cubic correction (`F(u) = -λu³`)
- **Policy gradient** → gradient-squared (`F(u) = λ(∇u)²`)

### 2. Classify Singularity

The noise `ξ` has Hölder regularity -(d/2 + 1). Heat kernel integration gains +2 in regularity. Products lose regularity. If the resulting regularity is negative, the solution is a distribution and products are ill-defined → **renormalization needed**.

### 3. Build the Regularity Structure

Construct the graded vector space `T` of symbols:
```
T = span{1, X_i, Ξ, I(Ξ), I(Ξ)·I(Ξ), I(Ξ²), ...}
```
with regularity exponents `A = {0, 1, ..., -(d/2+1), -(d/2+1)+2, ...}`. The structure group `G` acts on `T` to maintain consistency across base points.

### 4. Renormalize via Wick Ordering

Replace divergent products with Wick-ordered ones:
```
u²  →  :u²: = u² - ⟨u²⟩
u³  →  :u³: = u³ - 3⟨u²⟩u
```
This is equivalent to BPHZ renormalization in Hairer's framework.

### 5. Analyze RG Flow

The renormalization group describes how learning dynamics change under coarse-graining:
- **β-function**: `dλ/d(ln scale)` — how coupling constants flow
- **Fixed points**: scale-invariant dynamics (universal learning behavior)
- **Wilson-Fisher fixed point** at `λ* = ε/b`: governs critical learning

### 6. Classify into Universal Classes

Different agent architectures that share the same RG fixed point exhibit the same large-scale behavior, regardless of implementation details.

## The Math

### Singular SPDEs

The general form is:
```
∂ₜu = Lu + F(u, ∇u) + σξ
```

The linear stochastic heat equation (SHE) `∂ₜu = Δu + ξ` has solution regularity:
- d=1: `u ∈ C^{-1/2-ε}` (barely not a function)
- d=2: `u ∈ C^{-1-ε}` (a distribution)

After heat kernel convolution, noise gains +2 regularity. But nonlinear terms like `u²` or `(∇u)²` require the product of two distributions, which is undefined.

### Hairer's Regularity Structures

A regularity structure `(T, A, G)` generalizes Taylor expansion to distributions:

- **T** is a graded vector space: `T = ⊕_{α ∈ A} T_α`
- **A ⊂ ℝ** is the index set of regularity exponents
- **G** is the structure group acting on T

The **reconstruction operator** `R` maps modelled distributions (elements of T) to honest distributions. The fixed-point equation `U = S(LU + F(U))` is solved in the space of modelled distributions, where products are well-defined.

### Wick Ordering

For a centered Gaussian variable `u` with variance `σ²`:
```
:uⁿ: = Hₙ(u/σ) · σⁿ
```
where `Hₙ` is the n-th (probabilist's) Hermite polynomial:
```
H₀ = 1,  H₁ = x,  H₂ = x² - 1,  H₃ = x³ - 3x,  H₄ = x⁴ - 6x² + 3
```

### Renormalization Group

The β-function for Φ⁴ in d < 4 dimensions:
```
β(λ) = (4-d)λ - aλ²    (super-renormalizable)
```

Fixed points:
- λ* = 0 (Gaussian): trivial dynamics
- λ* = (4-d)/a (Wilson-Fisher): critical learning

The ε-expansion (`ε = 4 - d`) gives:
```
β(λ) = ελ - bλ² + cλ³
```

### KPZ Scaling

The KPZ equation exhibits anomalous scaling:
- Height fluctuations: `h ~ t^{1/3}` in 1d (not `t^{1/2}`)
- Tracy-Widom distribution describes the fluctuations
- This governs competitive agent learning dynamics

## Project Structure

```
src/
├── lib.rs           # Crate root, module declarations, re-exports
├── spde/mod.rs      # SPDE formulation, white noise, singularity classification
├── regularity/mod.rs # Regularity structures: symbols, models, reconstruction
├── wick/mod.rs      # Wick ordering, Hermite polynomials, renormalization
├── rg/mod.rs        # Renormalization group: β-functions, fixed points, RG flow
├── universal/mod.rs  # Universal classes: KPZ, Allen-Cahn, Φ⁴
├── holders/mod.rs   # Hölder regularity of belief trajectories
└── plato/mod.rs     # PLATO agent type → SPDE classification
```

## License

MIT
