//! # lau-singular-spde
//!
//! Singular stochastic PDE formulation of agent learning dynamics.
//!
//! Implements Hairer's regularity structures framework applied to multi-agent
//! learning dynamics, showing that certain agent update rules are singular SPDEs
//! requiring renormalization (Wick ordering).
//!
//! ## Core Concepts
//!
//! - **Stochastic PDE formulation**: Agent beliefs evolve as solutions to SPDEs
//! - **Regularity structures**: Hairer's framework for making sense of singular SPDEs
//! - **Wick ordering**: Renormalization of divergent products in agent dynamics
//! - **Renormalization group**: Coarse-graining flow on agent dynamics
//! - **Universal classes**: KPZ-like, Allen-Cahn, Φ⁴ critical behavior
//!
//! ## Universal Agent Learning Classes
//!
//! | Class | SPDE | Agent Behavior |
//! |-------|------|----------------|
//! | KPZ | ∂ₜh = ν∇²h + λ(∇h)² + ξ | Growth dynamics |
//! | Allen-Cahn | ∂ₜφ = Δφ + φ - φ³ + ξ | Phase transitions |
//! | Φ⁴ | ∂ₜφ = Δφ - m²φ - λφ³ + ξ | Critical learning |

pub mod spde;
pub mod regularity;
pub mod wick;
pub mod rg;
pub mod universal;
pub mod holders;
pub mod plato;

pub use spde::*;
pub use regularity::*;
pub use wick::*;
pub use rg::*;
pub use universal::*;
pub use holders::*;
pub use plato::*;
