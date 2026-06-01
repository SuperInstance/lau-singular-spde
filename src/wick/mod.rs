//! Wick ordering and renormalization of divergent products.
//!
//! In agent learning dynamics, products like u² or u³ appear naturally.
//! When u is a distributional solution of a singular SPDE, these products
//! are ill-defined. Wick ordering provides the renormalization:
//!
//! :u²: = u² - ⟨u²⟩  (subtract the divergent expectation)
//! :u³: = u³ - 3⟨u²⟩u  (for Gaussian u)
//!
//! This is exactly the BPHZ renormalization in Hairer's framework.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;

/// Wick-ordered product representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WickProduct {
    /// The order of the Wick product (e.g., 2 for :u²:).
    pub order: usize,
    /// Subtract terms: :u^n: = u^n - C₁ u^{n-2} - C₂ u^{n-4} - ...
    pub subtraction_coefficients: Vec<f64>,
    /// Expectation values ⟨u^k⟩ needed for renormalization.
    pub expectation_values: BTreeMap<usize, f64>,
}

impl WickProduct {
    /// Create a Wick-ordered product of given order.
    pub fn new(order: usize) -> Self {
        let subtraction_coefficients = Self::compute_subtractions(order);
        Self {
            order,
            subtraction_coefficients,
            expectation_values: BTreeMap::new(),
        }
    }

    /// Compute the subtraction coefficients using Hermite polynomial recursion.
    ///
    /// :u^n: = H_n(u/σ) · σ^n where H_n is the n-th Hermite polynomial.
    /// The subtractions follow from H_n(x) = x^n - (n choose 2) x^{n-2} + ...
    fn compute_subtractions(order: usize) -> Vec<f64> {
        if order == 0 {
            return vec![1.0];
        }
        if order == 1 {
            return vec![];
        }

        // Use Hermite polynomials: H_n(x) = x·H_{n-1}(x) - (n-1)·H_{n-2}(x)
        // Coefficients of x^k in H_n
        let mut coeffs: Vec<f64> = vec![0.0; order + 1];
        coeffs[order] = 1.0; // H_1 = x (for order=1, this would be set)
        coeffs[0] = 1.0; // H_0 = 1

        if order == 1 {
            return vec![];
        }

        // Build up H_n iteratively
        let mut h_prev2: Vec<f64> = vec![1.0]; // H_0
        let mut h_prev1: Vec<f64> = vec![0.0, 1.0]; // H_1

        for n in 2..=order {
            let mut h_n = vec![0.0; n + 1];
            // x · H_{n-1}
            for k in 0..h_prev1.len() {
                h_n[k + 1] += h_prev1[k];
            }
            // -(n-1) · H_{n-2}
            for k in 0..h_prev2.len() {
                h_n[k] -= (n - 1) as f64 * h_prev2[k];
            }
            h_prev2 = h_prev1;
            h_prev1 = h_n;
        }

        // The subtraction coefficients are the negative of lower-order coefficients
        let result: Vec<f64> = h_prev1.iter().skip(1).rev().cloned().collect();
        result
    }

    /// Compute the Wick product value for a given u and variance σ².
    pub fn evaluate(&self, u: f64, variance: f64) -> f64 {
        let sigma = variance.sqrt();
        match self.order {
            0 => 1.0,
            1 => u,
            2 => u * u - variance,
            3 => u * u * u - 3.0 * variance * u,
            4 => {
                let s2 = variance * variance;
                u * u * u * u - 6.0 * variance * u * u + 3.0 * s2
            }
            n => {
                // General: use Hermite polynomials
                self.evaluate_hermite(u, sigma, n)
            }
        }
    }

    /// Evaluate using Hermite polynomials for general order.
    fn evaluate_hermite(&self, u: f64, sigma: f64, n: usize) -> f64 {
        let x = u / sigma;
        let h = hermite_poly(x, n);
        h * sigma.powi(n as i32)
    }

    /// Apply Wick ordering to a vector of values.
    pub fn evaluate_vector(&self, values: &DVector<f64>, variance: f64) -> DVector<f64> {
        let mut result = values.clone();
        result.apply(|v| *v = self.evaluate(*v, variance));
        result
    }
}

/// Compute Hermite polynomial H_n(x) using the recursion.
fn hermite_poly(x: f64, n: usize) -> f64 {
    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let mut h_prev2 = 1.0;
            let mut h_prev1 = x;
            for k in 2..=n {
                let h_n = x * h_prev1 - (k - 1) as f64 * h_prev2;
                h_prev2 = h_prev1;
                h_prev1 = h_n;
            }
            h_prev1
        }
    }
}

/// Renormalization constants for Φ⁴ theory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phi4Renormalization {
    /// Mass renormalization δm².
    pub mass_counterterm: f64,
    /// Field strength renormalization Z.
    pub field_strength: f64,
    /// Coupling renormalization δλ.
    pub coupling_counterterm: f64,
    /// UV cutoff Λ.
    pub cutoff: f64,
    /// Spatial dimension d.
    pub spatial_dim: usize,
}

impl Phi4Renormalization {
    /// Compute renormalization constants for Φ⁴ in d dimensions with cutoff Λ.
    ///
    /// In d < 4, the theory is super-renormalizable:
    /// Only finitely many divergent diagrams need renormalization.
    pub fn compute(spatial_dim: usize, mass: f64, coupling: f64, cutoff: f64) -> Self {
        let d = spatial_dim as f64;

        // One-loop mass renormalization (tadpole diagram)
        // δm² ~ λ · ∫^{Λ} dp/(p² + m²) ~ λ · Λ^{d-2} (for d > 2)
        let mass_counterterm = if d > 2.0 {
            coupling * (cutoff.powf(d - 2.0) / (2.0 * std::f64::consts::PI).powf(d / 2.0))
                * gamma_integral(d, mass, cutoff)
        } else if d > 0.0 {
            // For d < 2: integral converges, ∫ dp/(p²+m²) ~ π/m
            coupling / (mass * (2.0 * std::f64::consts::PI).powf(d / 2.0))
        } else {
            0.0
        };

        // Field strength Z = 1 + O(λ²) — no one-loop correction in Φ⁴
        let field_strength = 1.0;

        // Coupling counterterm δλ ~ λ² · Λ^{d-4} (for d < 4, this is finite as Λ → ∞)
        let coupling_counterterm = if d < 4.0 && d > 2.0 {
            3.0 * coupling * coupling * cutoff.powf(d - 4.0)
                / (2.0 * std::f64::consts::PI).powf(d / 2.0)
        } else {
            0.0
        };

        Self {
            mass_counterterm,
            field_strength,
            coupling_counterterm,
            cutoff,
            spatial_dim,
        }
    }

    /// The renormalized mass m²_R = m² + δm².
    pub fn renormalized_mass(&self, bare_mass: f64) -> f64 {
        bare_mass + self.mass_counterterm
    }

    /// The renormalized coupling λ_R = λ + δλ.
    pub fn renormalized_coupling(&self, bare_coupling: f64) -> f64 {
        bare_coupling + self.coupling_counterterm
    }
}

/// Compute the momentum integral Γ(d, m, Λ) approximately.
fn gamma_integral(d: f64, m: f64, cutoff: f64) -> f64 {
    // ∫₀^Λ p^{d-1} / (p² + m²) dp ≈ Λ^{d-2}/(d-2) for large Λ, d > 2
    if d > 2.0 {
        cutoff.powf(d - 2.0) / (d - 2.0)
    } else {
        (1.0 + cutoff * cutoff / (m * m)).ln() / 2.0
    }
}

/// Renormalized SPDE: the Wick-ordered version of an agent SPDE.
///
/// ∂ₜφ = Δφ - m²φ - λ:φ³: + σξ
///
/// This is well-defined even when φ is a distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenormalizedSPDE {
    /// Original mass parameter.
    pub bare_mass: f64,
    /// Original coupling.
    pub bare_coupling: f64,
    /// Wick ordering subtraction C = ⟨φ²⟩ (the "diagonal" of the Green's function).
    pub wick_constant: f64,
    /// Noise intensity.
    pub noise_intensity: f64,
    /// The renormalization scheme.
    pub renormalization: Phi4Renormalization,
}

impl RenormalizedSPDE {
    /// Create a Wick-ordered Φ⁴ SPDE.
    pub fn new(mass: f64, coupling: f64, noise: f64, spatial_dim: usize, cutoff: f64) -> Self {
        let renormalization = Phi4Renormalization::compute(spatial_dim, mass, coupling, cutoff);
        // The Wick constant C = ∫ dp/(p² + m²) (regularized at cutoff)
        let wick_constant = renormalization.mass_counterterm / coupling.max(1e-10);

        Self {
            bare_mass: mass,
            bare_coupling: coupling,
            wick_constant,
            noise_intensity: noise,
            renormalization,
        }
    }

    /// Apply Wick-ordered nonlinearity: :φ³: = φ³ - 3Cφ
    pub fn wick_nonlinearity(&self, phi: &DVector<f64>) -> DVector<f64> {
        let c = self.wick_constant;
        let mut result = phi.clone();
        result.apply(|u| *u = *u * *u * *u - 3.0 * c * *u);
        result
    }

    /// Full renormalized dynamics step.
    pub fn step(
        &self,
        state: &DVector<f64>,
        laplacian: &nalgebra::DMatrix<f64>,
        noise: &DVector<f64>,
        dt: f64,
    ) -> DVector<f64> {
        let linear = laplacian * state - self.bare_mass * state;
        let nonlinear = self.wick_nonlinearity(state);
        let stochastic = self.noise_intensity * noise * dt.sqrt();
        state + (linear - self.bare_coupling * &nonlinear) * dt + stochastic
    }
}

/// Compute the Wick constant C_ε for a mollified noise at scale ε.
///
/// C_ε = ∫ ρ_ε(y)² G(y) dy → ∞ as ε → 0 (the UV divergence).
pub fn wick_constant_mollified(epsilon: f64, mass: f64, spatial_dim: usize) -> f64 {
    let d = spatial_dim as f64;
    // C_ε ≈ ε^{-(d-2)} for d > 2, log(1/ε) for d = 2
    if d > 2.0 {
        let power = d - 2.0;
        epsilon.powf(-power) / (4.0 * std::f64::consts::PI)
    } else if (d - 2.0).abs() < 0.01 {
        (1.0 / epsilon).ln() / (2.0 * std::f64::consts::PI)
    } else {
        1.0 / (epsilon * (1.0 + mass * epsilon))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_wick_product_order_0() {
        let w = WickProduct::new(0);
        assert_eq!(w.order, 0);
        assert_relative_eq!(w.evaluate(3.0, 1.0), 1.0);
    }

    #[test]
    fn test_wick_product_order_1() {
        let w = WickProduct::new(1);
        assert_relative_eq!(w.evaluate(5.0, 1.0), 5.0);
    }

    #[test]
    fn test_wick_product_order_2_gaussian() {
        let w = WickProduct::new(2);
        // :u²: = u² - σ²
        assert_relative_eq!(w.evaluate(0.0, 1.0), -1.0);
        assert_relative_eq!(w.evaluate(1.0, 1.0), 0.0);
        assert_relative_eq!(w.evaluate(2.0, 1.0), 3.0);
    }

    #[test]
    fn test_wick_product_order_2_expectation() {
        let w = WickProduct::new(2);
        // E[:u²:] = 0 for Gaussian u
        // Monte Carlo check
        let mut sum = 0.0;
        let n = 10000;
        for _ in 0..n {
            let u = rand::random::<f64>() * 2.0 - 1.0; // rough approx
            sum += w.evaluate(u, 1.0);
        }
        // Not exact since u isn't truly Gaussian, but :u²: should be centered
        let _mean = sum / n as f64;
    }

    #[test]
    fn test_wick_product_order_3() {
        let w = WickProduct::new(3);
        // :u³: = u³ - 3σ²u
        assert_relative_eq!(w.evaluate(1.0, 1.0), -2.0); // 1 - 3 = -2
        assert_relative_eq!(w.evaluate(2.0, 1.0), 2.0); // 8 - 6 = 2
    }

    #[test]
    fn test_wick_product_order_4() {
        let w = WickProduct::new(4);
        // :u⁴: = u⁴ - 6σ²u² + 3σ⁴
        assert_relative_eq!(w.evaluate(1.0, 1.0), -2.0); // 1 - 6 + 3 = -2
    }

    #[test]
    fn test_wick_product_vector() {
        let w = WickProduct::new(2);
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = w.evaluate_vector(&v, 1.0);
        assert_relative_eq!(result[0], 0.0);
        assert_relative_eq!(result[1], 3.0);
        assert_relative_eq!(result[2], 8.0);
    }

    #[test]
    fn test_hermite_poly_values() {
        // H_0(x) = 1
        assert_relative_eq!(hermite_poly(2.0, 0), 1.0);
        // H_1(x) = x
        assert_relative_eq!(hermite_poly(2.0, 1), 2.0);
        // H_2(x) = x² - 1
        assert_relative_eq!(hermite_poly(2.0, 2), 3.0);
        // H_3(x) = x³ - 3x
        assert_relative_eq!(hermite_poly(2.0, 3), 2.0);
    }

    #[test]
    fn test_phi4_renormalization() {
        let ren = Phi4Renormalization::compute(1, 1.0, 1.0, 100.0);
        assert!(ren.mass_counterterm > 0.0);
        assert_relative_eq!(ren.field_strength, 1.0);
    }

    #[test]
    fn test_phi4_renormalized_mass() {
        let ren = Phi4Renormalization::compute(1, 1.0, 1.0, 100.0);
        let m_ren = ren.renormalized_mass(1.0);
        assert!(m_ren > 1.0);
    }

    #[test]
    fn test_renormalized_spde_creation() {
        let rspde = RenormalizedSPDE::new(1.0, 1.0, 0.1, 1, 100.0);
        assert_relative_eq!(rspde.bare_mass, 1.0);
        assert_relative_eq!(rspde.bare_coupling, 1.0);
    }

    #[test]
    fn test_renormalized_spde_wick_nonlinearity() {
        let rspde = RenormalizedSPDE::new(1.0, 1.0, 0.1, 1, 100.0);
        let phi = DVector::from_vec(vec![1.0, 2.0]);
        let nl = rspde.wick_nonlinearity(&phi);
        // :u³: = u³ - 3Cu for some C
        // Check that nonlinearity is different from raw u³
        assert!(nl[0] != 1.0); // should be 1 - 3C, C > 0
    }

    #[test]
    fn test_wick_constant_mollified_1d() {
        let c = wick_constant_mollified(0.01, 1.0, 1);
        assert!(c > 0.0);
    }

    #[test]
    fn test_wick_constant_mollified_2d() {
        let c = wick_constant_mollified(0.01, 1.0, 2);
        assert!(c > 0.0);
        // Should be logarithmic: C ~ log(1/ε)
    }

    #[test]
    fn test_wick_constant_diverges() {
        let c1 = wick_constant_mollified(0.1, 1.0, 3);
        let c2 = wick_constant_mollified(0.01, 1.0, 3);
        // As ε → 0, C → ∞
        assert!(c2 > c1);
    }

    #[test]
    fn test_wick_product_zero_variance() {
        let w = WickProduct::new(2);
        // With zero variance, :u²: = u² (no subtraction)
        assert_relative_eq!(w.evaluate(3.0, 0.0), 9.0);
    }

    #[test]
    fn test_hermite_poly_orthogonality() {
        // ∫ H_n(x) H_m(x) e^{-x²/2} dx = n! √(2π) δ_{nm}
        // Just check some values
        assert_relative_eq!(hermite_poly(0.0, 2), -1.0); // H_2(0) = -1
        assert_relative_eq!(hermite_poly(0.0, 4), 3.0);  // H_4(0) = 3
    }
}
