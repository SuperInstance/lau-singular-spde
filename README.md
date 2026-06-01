# lau-singular-spde

A Rust library for **singular stochastic PDEs** — Hairer's regularity structures, Wick ordering, renormalization group flow, and universal agent learning dynamics.

## What This Does

This crate implements the mathematical machinery for studying agent learning dynamics that are governed by *singular* stochastic partial differential equations — equations where classical solutions don't exist and renormalization is required:

- **SPDE formulation** — Agent beliefs evolve as `∂ₜu = Lu + F(u) + ξ` where `ξ` is space-time white noise
- **Singularity classification** — Regular, Critical, or Singular based on regularity exponents
- **Regularity structures** — Hairer's framework: graded symbol spaces, model distributions, reconstruction operator, structure groups
- **Wick ordering** — Renormalization of divergent products: `:u²: = u² - ⟨u²⟩`, `:u³: = u³ - 3⟨u²⟩u`
- **Renormalization group** — Beta functions, fixed points (Gaussian, Wilson-Fisher), critical exponents, RG flow
- **Universal classes** — KPZ, Allen-Cahn, Φ⁴, and diffusive universality classes with their SPDEs and scaling behavior
- **Hölder regularity** — Regularity exponents for SPDE solutions, determining when renormalization is needed
- **PLATO classification** — Maps agent types (gradient descent, natural gradient, actor-critic, etc.) to their SPDE classification and renormalization requirements
- **Numerical solvers** — Euler-Maruyama integration, energy functionals, trajectory computation

## Key Idea

Certain agent learning rules, when viewed in the continuum limit, are *singular SPDEs* — the nonlinearity `F(u)` involves products of distributions that are mathematically ill-defined. Hairer's regularity structures provide a rigorous framework for making sense of these equations.

The library classifies agent dynamics into **universality classes** (analogous to statistical mechanics): different agent architectures can have the *same* large-scale behavior because they flow to the same fixed point of the renormalization group. KPZ-like dynamics describe competitive learning, Allen-Cahn describes phase transitions in agent populations, and Φ⁴ describes critical learning at the boundary between phases.

Wick ordering is the key renormalization trick: replace `u³` with `:u³: = u³ - 3Cu` where `C = ⟨u²⟩` is the divergent expectation that gets subtracted. This is exactly BPHZ renormalization in Hairer's framework.

## Install

```toml
[dependencies]
lau-singular-spde = "0.1"
```

Requires **Rust 2021 edition**.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Matrices, vectors, linear algebra |
| `serde` | Serialization of SPDEs, models, results |
| `rand` / `rand_distr` | White noise generation |
| `thiserror` | Error types |

## Quick Start

### Define and classify an SPDE

```rust
use lau_singular_spde::spde::{AgentSPDE, NonlinearityType, classify_singularity, SingularityClass};
use nalgebra::DMatrix;

// Φ⁴ equation in 1D: ∂ₜφ = Δφ - m²φ - λφ³ + ξ
let spde = AgentSPDE {
    dimension: 64,
    spatial_dim: 1,
    diffusion: DMatrix::identity(64, 64),
    nonlinearity: NonlinearityType::Cubic,
    noise_intensity: 0.5,
    mass: 1.0,
    coupling: 1.0,
};

let singularity = classify_singularity(&spde, 1);
assert_eq!(singularity, SingularityClass::Singular);
```

### Build a regularity structure

```rust
use lau_singular_spde::regularity::{Model, Symbol, build_structure};

// Build the canonical model with symbol hierarchy
let model = Model::build(&spde, 64, 3);

// Symbols: Ξ → I(Ξ) → I(Ξ)·I(Ξ) → I(I(Ξ)·I(Ξ)) → ...
let xi = Symbol::xi(1);       // regularity -1.5
let i_xi = Symbol::i_xi(&xi); // regularity 0.5

// Reconstruct a modelled distribution at a point
use lau_singular_spde::regularity::ModelledDistribution;
let md = ModelledDistribution::new(0.5)
    .with(&i_xi, 1.0);
let value = model.reconstruct(&md, 0);
```

### Wick ordering

```rust
use lau_singular_spde::wick::{WickProduct, RenormalizedSPDE, Phi4Renormalization};
use nalgebra::DVector;

// Wick-ordered products: :u²: = u² - σ², :u³: = u³ - 3σ²u
let w2 = WickProduct::new(2);
assert_eq!(w2.evaluate(0.0, 1.0), -1.0); // :0²: = 0 - 1 = -1
assert_eq!(w2.evaluate(1.0, 1.0), 0.0);  // :1²: = 1 - 1 = 0

let w3 = WickProduct::new(3);
assert_eq!(w3.evaluate(1.0, 1.0), -2.0); // :1³: = 1 - 3 = -2

// Create a renormalized Φ⁴ SPDE
let renorm = RenormalizedSPDE::new(1.0, 1.0, 0.5, 1, 100.0);
let phi = DVector::from_element(64, 0.1);
let wick_nl = renorm.wick_nonlinearity(&phi);
// :φ³: = φ³ - 3Cφ where C is the Wick constant
```

### Renormalization group flow

```rust
use lau_singular_spde::rg::{BetaFunction, FixedPoint};

// Φ⁴ beta function in d=3 (super-renormalizable)
let beta = BetaFunction::phi4_super_renormalizable(3);
let fps = beta.fixed_points(); // [0.0 (Gaussian), ε/3 (Wilson-Fisher)]

// Wilson-Fisher fixed point
let wf = FixedPoint::wilson_fisher(1.0); // ε = 4-d = 1
println!("WF coupling: λ* = {}", wf.couplings.coupling);
println!("Critical exponent: ν = {}", wf.critical_exponents[0]);
```

### Universal classes

```rust
use lau_singular_spde::universal::{KPZDynamics, AllenCahnDynamics, classify_universal};

// KPZ dynamics in 1D
let kpz = KPZDynamics::new(1.0, 1.0, 0.5, 1);
println!("Singularity: {}", kpz.singularity());     // Singular
println!("Scaling exponent: {}", kpz.scaling_exponent()); // 1/3

// Simulate
let trajectory = kpz.simulate(100, 10.0, 0.001, 1000);
```

### PLATO agent classification

```rust
use lau_singular_spde::plato::{PlatoAgentType, classify_agent};

let result = classify_agent(&PlatoAgentType::ActorCritic, 1);
println!("Needs renormalization: {}", result.needs_renormalization);
println!("Universal class: {}", result.universal_class);
println!("Wick order: {}", result.wick_order);
```

## API Reference

### `spde` — SPDE Formulation

| Item | Description |
|------|-------------|
| `AgentSPDE` | SPDE with `dimension`, `spatial_dim`, `diffusion` matrix, `nonlinearity`, `noise_intensity`, `mass`, `coupling` |
| `NonlinearityType` | `Linear`, `Quadratic`, `Cubic`, `GradientSquared`, `Polynomial(n)` |
| `SingularityClass` | `Regular`, `Critical`, `Singular` |
| `WhiteNoise` | Space-time white noise on a grid |
| `classify_singularity` | Determines singularity from SPDE parameters |
| `discrete_laplacian` | Finite-difference Laplacian with periodic BCs |
| `euler_maruyama_step` | One step of the Euler-Maruyama scheme |
| `solve_spde` | Full trajectory over multiple time steps |
| `energy_functional` | Allen-Cahn energy `E[u] = ∫(½|∇u|² + V(u))dx` |

### `regularity` — Regularity Structures

| Item | Description |
|------|-------------|
| `Symbol` | Abstract basis element with `name`, `regularity`, `poly_degree` |
| `ModelledDistribution` | Element of T_α with symbol→coefficient map |
| `Model` | Concrete realization: assigns grid values to symbols, `reconstruct()` operator |
| `StructureGroup` | Γ_{xy}: maps distributions between base points |
| `build_structure` | Constructs full symbol hierarchy for an SPDE |

### `wick` — Wick Ordering & Renormalization

| Item | Description |
|------|-------------|
| `WickProduct` | Wick-ordered monomial `:uⁿ:` via Hermite polynomials |
| `Phi4Renormalization` | Mass, field-strength, and coupling counterterms for Φ⁴ |
| `RenormalizedSPDE` | Full Wick-ordered Φ⁴ dynamics with `step()` |
| `wick_constant_mollified` | UV-divergent constant `C_ε` for mollified noise |

### `rg` — Renormalization Group

| Item | Description |
|------|-------------|
| `BetaFunction` | `Phi4SuperRenorm`, `WilsonFisher`, `KPZ`, `Custom` — RG flow of coupling |
| `CouplingSpace` | (mass², coupling, field_strength) at a fixed point |
| `FixedPoint` | Coupling values, stability, critical exponents |
| `FixedPoint::gaussian(d)` | Free-theory fixed point |
| `FixedPoint::wilson_fisher(ε)` | Non-trivial FP at `λ* = ε/3` |

### `universal` — Universal Agent Learning Classes

| Class | SPDE | Agent Behavior |
|-------|------|----------------|
| `KPZ` | `∂ₜh = ν∇²h + λ(∇h)² + ξ` | Competitive growth, `t^{1/3}` scaling |
| `AllenCahn` | `∂ₜφ = Δφ + φ - φ³ + ξ` | Bistable phase transitions |
| `Phi4` | `∂ₜφ = Δφ - m²φ - λ:φ³: + ξ` | Critical learning at phase boundary |
| `Diffusive` | `∂ₜu = Δu - m²u + ξ` | Gaussian (trivial) fixed point |
| `MeanField` | All agents see average | Classical game theory limit |

### `holders` — Hölder Regularity

| Item | Description |
|------|-------------|
| `HolderRegularity` | Exponent `α`, function/distribution classification, description |
| `holder_exponent` | Theoretical Hölder exponent for SPDE solutions |
| `needs_renormalization` | Boolean check based on regularity |

### `plato` — PLATO Agent Classification

| Agent Type | Typical Classification |
|-----------|----------------------|
| `GradientDescent` | Linear / Diffusive |
| `NaturalGradient` | Quadratic / KPZ-like |
| `ActorCritic` | Cubic / Φ⁴-like |
| `MetaLearner` | Depends on adaptation rule |
| `Ensemble` | Coupled system, higher-dimension SPDE |
| `Attention` | Nonlocal operator, exotic regularity |

## How It Works

### Singularity classification

Space-time white noise `ξ` in `d` spatial dimensions has regularity `α = -(d/2 + 1)`. The heat kernel smooths by +2, so the linear solution has regularity `α + 2`. When the nonlinearity requires products (e.g., `u²` or `u³`), the product regularity is the sum of the factors'. If this falls below zero, the product is ill-defined and the SPDE is singular.

### Regularity structure construction

The symbol hierarchy is built iteratively:
1. Start with noise symbol `Ξ` (regularity `-(d/2+1)`)
2. Integrate through heat kernel: `I(Ξ)` gains +2 regularity
3. Form products: `I(Ξ)·I(Ξ)`, `I(Ξ)³`
4. Integrate again: `I(I(Ξ)·I(Ξ))`, etc.

The model `Π` maps abstract symbols to concrete grid values. The reconstruction operator `R` converts modelled distributions back to real-valued functions.

### Wick ordering via Hermite polynomials

The Wick product `:uⁿ:` equals `Hₙ(u/σ)·σⁿ` where `Hₙ` is the n-th probabilist's Hermite polynomial:

```
H₀ = 1,  H₁ = x,  H₂ = x²-1,  H₃ = x³-3x,  H₄ = x⁴-6x²+3
```

Key property: `E[:uⁿ:] = 0` for Gaussian `u` and `n ≥ 1`.

### RG flow and fixed points

The beta function `β(λ) = dλ/dt` governs how the coupling changes under coarse-graining. Fixed points `β(λ*) = 0` correspond to scale-invariant theories. For Φ⁴ in `d = 4-ε` dimensions:

```
β(λ) = ελ - 3λ²  →  λ* = 0 (Gaussian) or λ* = ε/3 (Wilson-Fisher)
```

Critical exponents (correlation length, anomalous dimension) are eigenvalues of the linearized RG at the fixed point.

## The Math

**Stochastic heat equation** (the fundamental singular SPDE):

```
∂ₜu = Δu + ξ,  ξ = space-time white noise
```

In `d ≥ 2`, the solution `u` is a *distribution*, not a function. The heat kernel gains `+2` regularity, but noise has regularity `-(d/2+1)`, so the solution regularity is `-(d/2-1)`, which is negative for `d ≥ 2`.

**Hairer's regularity structures** (Fields Medal 2014): construct a triple `(T, A, G)` where:
- `T = ⊕_α T_α` is a graded space of "abstract noise polynomials"
- `A ⊂ R` is the set of regularity exponents
- `G` is the structure group acting on `T`

A *model* `(Π, Γ)` maps `T` to concrete distributions. The fixed-point problem `u = Gu + F(u)` is solved in the space of modelled distributions, then *reconstructed* to a distribution-valued solution.

**Wick ordering / BPHZ renormalization**: products like `u²` diverge. Replace with `:u²: = u² - C` where `C = E[u²]` is the (infinite) expectation, regularized at scale `ε` and renormalized as `ε → 0`.

**KPZ scaling**: in 1D, height fluctuations scale as `t^{1/3}` (not `t^{1/2}` as for diffusive processes). This is the signature of the KPZ universality class — Tracy-Widom fluctuations, `1/f` noise.

## License

MIT
