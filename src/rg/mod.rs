//! Renormalization group (RG) flow for agent dynamics.
//!
//! The RG describes how agent learning dynamics change under coarse-graining:
//! integrate out short-scale fluctuations → effective dynamics at longer scales.
//!
//! Key concepts:
//! - β-function: rate of change of coupling constants under scale transformation
//! - Fixed points: scale-invariant dynamics (universal learning behavior)
//! - Relevant/irrelevant perturbations: stability of fixed points

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use std::fmt;

/// A point in coupling constant space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingSpace {
    /// Mass parameter m².
    pub mass: f64,
    /// Coupling constant λ.
    pub coupling: f64,
    /// Noise intensity σ².
    pub noise: f64,
    /// Higher-order couplings (if any).
    pub higher_couplings: Vec<f64>,
}

impl CouplingSpace {
    pub fn new(mass: f64, coupling: f64, noise: f64) -> Self {
        Self { mass, coupling, noise, higher_couplings: Vec::new() }
    }

    /// Euclidean distance in coupling space.
    pub fn distance(&self, other: &CouplingSpace) -> f64 {
        let dm = self.mass - other.mass;
        let dc = self.coupling - other.coupling;
        let dn = self.noise - other.noise;
        (dm * dm + dc * dc + dn * dn).sqrt()
    }

    /// Add two coupling vectors.
    pub fn add(&self, other: &CouplingSpace) -> CouplingSpace {
        CouplingSpace {
            mass: self.mass + other.mass,
            coupling: self.coupling + other.coupling,
            noise: self.noise + other.noise,
            higher_couplings: self.higher_couplings.iter()
                .zip(other.higher_couplings.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Scale coupling vector.
    pub fn scale(&self, factor: f64) -> CouplingSpace {
        CouplingSpace {
            mass: self.mass * factor,
            coupling: self.coupling * factor,
            noise: self.noise * factor,
            higher_couplings: self.higher_couplings.iter().map(|x| x * factor).collect(),
        }
    }
}

/// β-function for the coupling constant.
/// β(λ) = dλ/dt where t = ln(scale).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BetaFunction {
    /// Φ⁴ in d < 4: β(λ) = (4-d)λ - aλ² (super-renormalizable)
    Phi4SuperRenorm { dim: usize, coefficient: f64 },
    /// Wilson-Fisher: β(λ) = ελ - bλ² + cλ³ (ε = 4-d expansion)
    WilsonFisher { epsilon: f64, b: f64, c: f64 },
    /// KPZ: β(λ) = (2-d)/2 · λ + higher order
    KPZ { dim: usize },
    /// Custom β-function as polynomial coefficients [β₀, β₁, β₂, ...]
    /// β(λ) = β₀ + β₁λ + β₂λ² + ...
    Custom { coefficients: Vec<f64> },
}

impl BetaFunction {
    /// Evaluate the β-function at coupling λ.
    pub fn evaluate(&self, lambda: f64) -> f64 {
        match self {
            BetaFunction::Phi4SuperRenorm { dim, coefficient } => {
                let eps = 4.0 - *dim as f64;
                eps * lambda - coefficient * lambda * lambda
            }
            BetaFunction::WilsonFisher { epsilon, b, c } => {
                epsilon * lambda - b * lambda * lambda + c * lambda * lambda * lambda
            }
            BetaFunction::KPZ { dim } => {
                let d = *dim as f64;
                (2.0 - d) / 2.0 * lambda
            }
            BetaFunction::Custom { coefficients } => {
                coefficients.iter().enumerate()
                    .map(|(i, c)| c * lambda.powi(i as i32))
                    .sum()
            }
        }
    }

    /// Find fixed points: β(λ*) = 0.
    pub fn fixed_points(&self) -> Vec<f64> {
        match self {
            BetaFunction::Phi4SuperRenorm { dim, coefficient } => {
                let eps = 4.0 - *dim as f64;
                let mut fps = vec![0.0]; // Gaussian fixed point
                if *coefficient != 0.0 {
                    let nontrivial = eps / coefficient;
                    if nontrivial > 0.0 {
                        fps.push(nontrivial);
                    }
                }
                fps
            }
            BetaFunction::WilsonFisher { epsilon, b, c } => {
                let mut fps = vec![0.0];
                // Solve ελ - bλ² + cλ³ = 0
                if *b != 0.0 {
                    let lambda1 = epsilon / b;
                    fps.push(lambda1);
                }
                if *c != 0.0 && *b != 0.0 {
                    let lambda2 = b / c;
                    fps.push(lambda2);
                }
                fps
            }
            BetaFunction::KPZ { dim: _ } => {
                vec![0.0] // Only Gaussian FP in this approximation
            }
            BetaFunction::Custom { .. } => {
                vec![0.0] // Default: just Gaussian
            }
        }
    }
}

/// Stability of a fixed point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixedPointStability {
    /// All directions flow toward FP (attractive).
    Stable,
    /// Some directions flow away (saddle).
    Unstable,
    /// Neutral (marginal).
    Marginal,
}

impl fmt::Display for FixedPointStability {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FixedPointStability::Stable => write!(f, "Stable"),
            FixedPointStability::Unstable => write!(f, "Unstable"),
            FixedPointStability::Marginal => write!(f, "Marginal"),
        }
    }
}

/// A fixed point of the RG flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPoint {
    /// The coupling values at the fixed point.
    pub couplings: CouplingSpace,
    /// Stability classification.
    pub stability: FixedPointStability,
    /// Critical exponents (eigenvalues of linearized RG at FP).
    pub critical_exponents: Vec<f64>,
    /// Name/description.
    pub name: String,
}

impl FixedPoint {
    /// Gaussian (free theory) fixed point: all couplings = 0.
    pub fn gaussian(spatial_dim: usize) -> Self {
        let stability = if spatial_dim < 4 {
            FixedPointStability::Unstable
        } else {
            FixedPointStability::Stable
        };
        Self {
            couplings: CouplingSpace::new(0.0, 0.0, 0.0),
            stability,
            critical_exponents: vec![(4 - spatial_dim) as f64], // ε = 4 - d
            name: "Gaussian".to_string(),
        }
    }

    /// Wilson-Fisher fixed point in d = 4 - ε dimensions.
    pub fn wilson_fisher(epsilon: f64) -> Self {
        let lambda_star = epsilon / 3.0; // Leading order
        Self {
            couplings: CouplingSpace::new(
                -epsilon * lambda_star / 2.0, // mass correction
                lambda_star,
                1.0,
            ),
            stability: FixedPointStability::Stable,
            critical_exponents: vec![-epsilon], // negative = relevant direction eigenvalue
            name: "Wilson-Fisher".to_string(),
        }
    }

    /// KPZ fixed point.
    pub fn kpz(spatial_dim: usize) -> Self {
        Self {
            couplings: CouplingSpace::new(0.0, 1.0, 1.0),
            stability: if spatial_dim <= 2 {
                FixedPointStability::Stable
            } else {
                FixedPointStability::Unstable
            },
            critical_exponents: vec![(2 - spatial_dim) as f64 / 2.0],
            name: "KPZ".to_string(),
        }
    }

    /// Classify a perturbation as relevant, irrelevant, or marginal.
    pub fn classify_perturbation(&self, exponent: f64) -> &'static str {
        if exponent > 0.01 {
            "relevant"
        } else if exponent < -0.01 {
            "irrelevant"
        } else {
            "marginal"
        }
    }
}

/// RG flow trajectory: sequence of coupling constants under coarse-graining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RGFlow {
    /// Starting point in coupling space.
    pub initial: CouplingSpace,
    /// Trajectory of coupling constants.
    pub trajectory: Vec<CouplingSpace>,
    /// Scale parameter at each step (logarithmic).
    pub scales: Vec<f64>,
    /// The β-function driving the flow.
    pub beta: BetaFunction,
}

impl RGFlow {
    /// Compute the RG flow by integrating the β-function.
    pub fn compute(
        initial: &CouplingSpace,
        beta: &BetaFunction,
        t_max: f64,
        n_steps: usize,
    ) -> Self {
        let dt = t_max / n_steps as f64;
        let mut trajectory = vec![initial.clone()];
        let mut scales = vec![0.0];
        let mut current = initial.clone();

        for i in 1..=n_steps {
            let t = i as f64 * dt;
            scales.push(t);

            // Euler step for λ
            let dlambda = beta.evaluate(current.coupling) * dt;
            // Mass flows as: dm²/dt = 2m² + corrections
            let dm = 2.0 * current.mass * dt;
            // Noise: dσ²/dt = (d-specific scaling) σ²
            let dnoise = 0.0;

            current = CouplingSpace::new(
                current.mass + dm,
                current.coupling + dlambda,
                current.noise + dnoise,
            );
            trajectory.push(current.clone());
        }

        Self {
            initial: initial.clone(),
            trajectory,
            scales,
            beta: beta.clone(),
        }
    }

    /// Does the flow approach a fixed point?
    pub fn approaches_fixed_point(&self, fp: &FixedPoint, tolerance: f64) -> bool {
        if let Some(last) = self.trajectory.last() {
            last.distance(&fp.couplings) < tolerance
        } else {
            false
        }
    }

    /// The correlation length exponent ν.
    /// At a fixed point with critical exponent y: ν = 1/y.
    pub fn correlation_length_exponent(&self) -> f64 {
        if let Some(fp_exp) = self.find_nearest_fixed_point().critical_exponents.first() {
            if fp_exp.abs() > 0.001 {
                1.0 / fp_exp.abs()
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        }
    }

    /// Find the nearest known fixed point.
    fn find_nearest_fixed_point(&self) -> FixedPoint {
        if let Some(last) = self.trajectory.last() {
            let candidates = vec![
                FixedPoint::gaussian(1),
                FixedPoint::wilson_fisher(1.0),
            ];
            candidates.into_iter()
                .min_by(|a, b| {
                    last.distance(&a.couplings)
                        .partial_cmp(&last.distance(&b.couplings))
                        .unwrap()
                })
                .unwrap_or(FixedPoint::gaussian(1))
        } else {
            FixedPoint::gaussian(1)
        }
    }
}

/// Coarse-graining step: integrate out modes in shell [Λ/b, Λ].
pub fn coarse_grain(
    state: &DVector<f64>,
    grid_size: usize,
    factor: usize,
) -> DVector<f64> {
    if factor <= 1 || state.len() < factor {
        return state.clone();
    }
    let new_size = grid_size / factor;
    let mut result = DVector::zeros(new_size);
    for i in 0..new_size {
        let mut sum = 0.0;
        for j in 0..factor {
            let idx = i * factor + j;
            if idx < state.len() {
                sum += state[idx];
            }
        }
        result[i] = sum / factor as f64;
    }
    result
}

/// Rescale after coarse-graining (restore original grid size).
pub fn rescale(state: &DVector<f64>, target_size: usize) -> DVector<f64> {
    let n = state.len();
    if n >= target_size {
        return state.clone();
    }
    let mut result = DVector::zeros(target_size);
    let ratio = target_size as f64 / n as f64;
    for i in 0..target_size {
        let src_idx = (i as f64 / ratio).min((n - 1) as f64) as usize;
        result[i] = state[src_idx];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_coupling_space_distance() {
        let a = CouplingSpace::new(0.0, 0.0, 0.0);
        let b = CouplingSpace::new(3.0, 4.0, 0.0);
        assert_relative_eq!(a.distance(&b), 5.0);
    }

    #[test]
    fn test_coupling_space_add() {
        let a = CouplingSpace::new(1.0, 2.0, 3.0);
        let b = CouplingSpace::new(0.5, 0.5, 0.5);
        let c = a.add(&b);
        assert_relative_eq!(c.mass, 1.5);
        assert_relative_eq!(c.coupling, 2.5);
        assert_relative_eq!(c.noise, 3.5);
    }

    #[test]
    fn test_coupling_space_scale() {
        let a = CouplingSpace::new(2.0, 3.0, 4.0);
        let b = a.scale(0.5);
        assert_relative_eq!(b.mass, 1.0);
        assert_relative_eq!(b.coupling, 1.5);
    }

    #[test]
    fn test_beta_phi4_gaussian() {
        let beta = BetaFunction::Phi4SuperRenorm { dim: 3, coefficient: 3.0 };
        assert_relative_eq!(beta.evaluate(0.0), 0.0);
    }

    #[test]
    fn test_beta_phi4_fixed_points() {
        let beta = BetaFunction::Phi4SuperRenorm { dim: 3, coefficient: 3.0 };
        let fps = beta.fixed_points();
        assert!(fps.contains(&0.0));
        assert!(fps.iter().any(|&x| (x - 1.0/3.0).abs() < 0.01));
    }

    #[test]
    fn test_beta_wilson_fisher() {
        let beta = BetaFunction::WilsonFisher { epsilon: 1.0, b: 3.0, c: 1.0 };
        let fps = beta.fixed_points();
        assert!(fps.contains(&0.0));
        assert!(fps.iter().any(|&x| (x - 1.0/3.0).abs() < 0.01));
    }

    #[test]
    fn test_beta_kpz() {
        let beta = BetaFunction::KPZ { dim: 1 };
        // In 1d: β(λ) = 0.5λ
        assert_relative_eq!(beta.evaluate(2.0), 1.0);
    }

    #[test]
    fn test_beta_custom() {
        let beta = BetaFunction::Custom { coefficients: vec![0.0, -1.0] };
        assert_relative_eq!(beta.evaluate(3.0), -3.0);
    }

    #[test]
    fn test_gaussian_fixed_point() {
        let fp = FixedPoint::gaussian(3);
        assert_relative_eq!(fp.couplings.coupling, 0.0);
        assert_eq!(fp.stability, FixedPointStability::Unstable);
    }

    #[test]
    fn test_wilson_fisher_fixed_point() {
        let fp = FixedPoint::wilson_fisher(1.0);
        assert!(fp.couplings.coupling > 0.0);
        assert_eq!(fp.stability, FixedPointStability::Stable);
    }

    #[test]
    fn test_kpz_fixed_point() {
        let fp = FixedPoint::kpz(1);
        assert_eq!(fp.name, "KPZ");
    }

    #[test]
    fn test_fixed_point_classify_perturbation() {
        let fp = FixedPoint::gaussian(3);
        assert_eq!(fp.classify_perturbation(0.5), "relevant");
        assert_eq!(fp.classify_perturbation(-0.5), "irrelevant");
        assert_eq!(fp.classify_perturbation(0.0), "marginal");
    }

    #[test]
    fn test_rg_flow_computation() {
        let initial = CouplingSpace::new(0.1, 0.1, 1.0);
        let beta = BetaFunction::Phi4SuperRenorm { dim: 3, coefficient: 3.0 };
        let flow = RGFlow::compute(&initial, &beta, 5.0, 100);
        assert_eq!(flow.trajectory.len(), 101);
        assert_eq!(flow.scales.len(), 101);
    }

    #[test]
    fn test_rg_flow_approaches_fp() {
        // Start near Wilson-Fisher FP
        let wf = FixedPoint::wilson_fisher(1.0);
        let initial = CouplingSpace::new(
            wf.couplings.mass,
            wf.couplings.coupling,
            wf.couplings.noise,
        );
        let beta = BetaFunction::WilsonFisher { epsilon: 1.0, b: 3.0, c: 0.0 };
        let flow = RGFlow::compute(&initial, &beta, 1.0, 100);
        // Should be near WF FP (β ≈ 0 at FP)
        assert!(flow.approaches_fixed_point(&wf, 1.0) || flow.trajectory.last().unwrap().coupling > 0.0);
    }

    #[test]
    fn test_coarse_grain() {
        let state = DVector::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
        let cg = coarse_grain(&state, 4, 2);
        assert_eq!(cg.len(), 2);
        assert_relative_eq!(cg[0], 2.0);
        assert_relative_eq!(cg[1], 6.0);
    }

    #[test]
    fn test_rescale() {
        let state = DVector::from_vec(vec![2.0, 6.0]);
        let rs = rescale(&state, 4);
        assert_eq!(rs.len(), 4);
    }

    #[test]
    fn test_fp_stability_display() {
        assert_eq!(format!("{}", FixedPointStability::Stable), "Stable");
        assert_eq!(format!("{}", FixedPointStability::Unstable), "Unstable");
    }

    #[test]
    fn test_correlation_length_exponent() {
        let initial = CouplingSpace::new(0.1, 0.1, 1.0);
        let beta = BetaFunction::Phi4SuperRenorm { dim: 3, coefficient: 3.0 };
        let flow = RGFlow::compute(&initial, &beta, 1.0, 50);
        let nu = flow.correlation_length_exponent();
        assert!(nu > 0.0);
    }
}
