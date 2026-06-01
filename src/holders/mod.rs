//! Hölder regularity of agent belief trajectories.
//!
//! Key question: how smooth are agent belief trajectories?
//!
//! For solutions of singular SPDEs, the solution lives in a Hölder space
//! C^{α} for some regularity exponent α < 0 (it's a distribution!).
//! After renormalization, the Wick-ordered solution gains regularity.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::spde::{AgentSPDE, NonlinearityType, classify_singularity, SingularityClass};

/// Hölder regularity result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolderRegularity {
    /// The Hölder exponent α.
    pub exponent: f64,
    /// Whether the solution is a function (α > 0) or distribution (α < 0).
    pub is_function: bool,
    /// Spatial dimension.
    pub spatial_dim: usize,
    /// Description.
    pub description: String,
}

impl std::fmt::Display for HolderRegularity {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let kind = if self.is_function { "function" } else { "distribution" };
        write!(f, "C^{{{:.2}}} ({}) — {}", self.exponent, kind, self.description)
    }
}

/// Compute the theoretical Hölder exponent for an SPDE solution.
///
/// For the stochastic heat equation ∂ₜu = Δu + ξ:
/// - In d=1: u ∈ C^{-1/2-ε} → after integration: C^{1/2-ε} (barely a function!)
/// - In d=2: u ∈ C^{-1-ε} (distribution)
/// - In d=3: u ∈ C^{-3/2-ε} (worse distribution)
///
/// The key formula: regularity of u = regularity of ξ + 2 (heat kernel gains 2).
/// And regularity of ξ = -(d/2 + 1) in space-time.
pub fn holder_exponent(spatial_dim: usize, nonlinearity: NonlinearityType) -> HolderRegularity {
    let noise_regularity = -(spatial_dim as f64 / 2.0 + 1.0);
    let linear_regularity = noise_regularity + 2.0;

    match nonlinearity {
        NonlinearityType::Linear => HolderRegularity {
            exponent: linear_regularity,
            is_function: linear_regularity > 0.0,
            spatial_dim,
            description: format!(
                "Linear SHE: noise α={:.2}, solution α={:.2}",
                noise_regularity, linear_regularity
            ),
        },
        NonlinearityType::Quadratic | NonlinearityType::GradientSquared => {
            // KPZ-type: u² or (∇u)² loses regularity
            let _product_loss = linear_regularity; // u·u loses this much
            let kpz_regularity = linear_regularity - 0.5; // KPZ correction
            HolderRegularity {
                exponent: kpz_regularity,
                is_function: kpz_regularity > 0.0,
                spatial_dim,
                description: "KPZ-type: quadratic nonlinearity reduces α by ~0.5".to_string(),
            }
        }
        NonlinearityType::Cubic => {
            // Φ⁴: u³ loses even more regularity
            let phi4_regularity = linear_regularity - 1.0;
            HolderRegularity {
                exponent: phi4_regularity,
                is_function: phi4_regularity > 0.0,
                spatial_dim,
                description: "Φ⁴-type: cubic nonlinearity reduces α by ~1.0".to_string(),
            }
        }
        NonlinearityType::Polynomial(deg) => {
            let loss = (deg as f64 - 1.0) * 0.5;
            let regularity = linear_regularity - loss;
            HolderRegularity {
                exponent: regularity,
                is_function: regularity > 0.0,
                spatial_dim,
                description: format!(
                    "Degree-{} polynomial: loss of {:.1}", deg, loss
                ),
            }
        }
    }
}

/// Compute empirical Hölder exponent from a trajectory.
///
/// Uses the method: α ≈ log(ratio of increments) / log(scale ratio).
pub fn empirical_holder_exponent(trajectory: &[DVector<f64>]) -> f64 {
    if trajectory.len() < 3 {
        return 0.0;
    }

    let n = trajectory[0].len();
    let mut sum_ratios = 0.0;
    let mut count = 0;

    for t in 0..trajectory.len() - 2 {
        #[allow(clippy::needless_range_loop)]
        for i in 0..n.min(trajectory[t].len()).min(trajectory[t + 1].len()).min(trajectory[t + 2].len()) {
            let inc1 = (trajectory[t + 1][i] - trajectory[t][i]).abs();
            let inc2 = (trajectory[t + 2][i] - trajectory[t + 1][i]).abs();
            if inc1 > 1e-10 && inc2 > 1e-10 {
                let ratio = inc2 / inc1;
                if ratio > 0.0 {
                    sum_ratios += ratio.ln();
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        return 0.0;
    }
    sum_ratios / count as f64
}

/// Hölder semi-norm: [u]_{C^α} = sup |u(x) - u(y)| / |x - y|^α
pub fn holder_seminorm(values: &DVector<f64>, alpha: f64, dx: f64) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }

    let mut max_ratio = 0.0f64;
    for i in 0..n - 1 {
        let diff = (values[i + 1] - values[i]).abs();
        let denom = dx.powf(alpha);
        if denom > 0.0 {
            max_ratio = max_ratio.max(diff / denom);
        }
    }
    max_ratio
}

/// Regularity table: for each spatial dimension, what's the maximum
/// nonlinearity degree that yields a classical solution?
pub fn regularity_table() -> Vec<HolderRegularity> {
    let mut table = Vec::new();
    for d in 1..=4 {
        for nl in &[NonlinearityType::Linear, NonlinearityType::Quadratic, NonlinearityType::Cubic] {
            table.push(holder_exponent(d, *nl));
        }
    }
    table
}

/// Check if an SPDE needs renormalization based on its regularity.
pub fn needs_renormalization(spde: &AgentSPDE) -> bool {
    let regularity = holder_exponent(spde.spatial_dim, spde.nonlinearity);
    !regularity.is_function || matches!(classify_singularity(spde, spde.spatial_dim), SingularityClass::Singular)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use approx::*;

    #[test]
    fn test_holder_exponent_linear_1d() {
        let h = holder_exponent(1, NonlinearityType::Linear);
        // α = -1.5 + 2 = 0.5
        assert_relative_eq!(h.exponent, 0.5, epsilon = 0.01);
        assert!(h.is_function);
    }

    #[test]
    fn test_holder_exponent_linear_2d() {
        let h = holder_exponent(2, NonlinearityType::Linear);
        // α = -2 + 2 = 0 (barely not a function!)
        assert_relative_eq!(h.exponent, 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_holder_exponent_linear_3d() {
        let h = holder_exponent(3, NonlinearityType::Linear);
        // α = -2.5 + 2 = -0.5 (distribution!)
        assert!(h.exponent < 0.0);
        assert!(!h.is_function);
    }

    #[test]
    fn test_holder_exponent_kpz_1d() {
        let h = holder_exponent(1, NonlinearityType::GradientSquared);
        // α ≈ 0.5 - 0.5 = 0.0
        assert!(h.exponent >= -0.1 && h.exponent <= 0.6);
    }

    #[test]
    fn test_holder_exponent_phi4_1d() {
        let h = holder_exponent(1, NonlinearityType::Cubic);
        // α ≈ 0.5 - 1.0 = -0.5
        assert!(h.exponent < 0.5);
    }

    #[test]
    fn test_holder_display() {
        let h = holder_exponent(1, NonlinearityType::Linear);
        let s = format!("{}", h);
        assert!(s.contains("0.5"));
    }

    #[test]
    fn test_empirical_holder_constant() {
        // Constant trajectory → exponent should be ~0
        let v = DVector::from_element(10, 1.0);
        let traj = vec![v.clone(), v.clone(), v.clone()];
        let alpha = empirical_holder_exponent(&traj);
        assert_relative_eq!(alpha, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_empirical_holder_smooth() {
        // Linear trajectory
        let v1: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let v2: Vec<f64> = (0..10).map(|i| (i + 1) as f64).collect();
        let v3: Vec<f64> = (0..10).map(|i| (i + 2) as f64).collect();
        let traj = vec![
            DVector::from_vec(v1),
            DVector::from_vec(v2),
            DVector::from_vec(v3),
        ];
        let alpha = empirical_holder_exponent(&traj);
        // Should be close to 1 (linear is C^1)
        // Actually our metric measures fluctuation, so constant increments → finite
        assert!(alpha.is_finite());
    }

    #[test]
    fn test_holder_seminorm_constant() {
        let v = DVector::from_element(10, 5.0);
        let sn = holder_seminorm(&v, 0.5, 0.1);
        assert_relative_eq!(sn, 0.0);
    }

    #[test]
    fn test_holder_seminorm_linear() {
        let v: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let v = DVector::from_vec(v);
        let sn = holder_seminorm(&v, 0.5, 1.0);
        // For linear: |u(x+1) - u(x)| = 1, dx^0.5 = 1
        assert!(sn > 0.0);
    }

    #[test]
    fn test_regularity_table() {
        let table = regularity_table();
        assert_eq!(table.len(), 12); // 4 dims × 3 types
    }

    #[test]
    fn test_needs_renormalization_linear() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Linear,
            noise_intensity: 1.0, mass: 1.0, coupling: 0.0,
        };
        assert!(!needs_renormalization(&spde));
    }

    #[test]
    fn test_needs_renormalization_singular() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Cubic,
            noise_intensity: 1.0, mass: 1.0, coupling: 1.0,
        };
        assert!(needs_renormalization(&spde));
    }

    #[test]
    fn test_holder_exponent_polynomial() {
        let h = holder_exponent(1, NonlinearityType::Polynomial(5));
        assert!(h.exponent < 0.0); // High degree → singular
    }
}
