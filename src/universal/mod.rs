//! Universal agent learning classes.
//!
//! Agent learning dynamics fall into universality classes analogous to
//! well-known SPDEs. The key insight: different agent architectures can
//! exhibit the SAME large-scale behavior (same fixed point of RG flow).
//!
//! | Class | SPDE | Agent Behavior | Singularity |
//! |-------|------|----------------|-------------|
//! | KPZ | ∂ₜh = ν∇²h + λ(∇h)² + ξ | Growth/competition | Singular in d≥1 |
//! | Allen-Cahn | ∂ₜφ = Δφ + φ - φ³ + ξ | Phase transition | Singular in d≥2 |
//! | Φ⁴ | ∂ₜφ = Δφ - m²φ - λ:φ³: + ξ | Critical learning | Singular in d≥1 |

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::spde::{
    AgentSPDE, NonlinearityType, SingularityClass,
    classify_singularity, discrete_laplacian, euler_maruyama_step, solve_spde,
    energy_functional,
};
use crate::rg::{FixedPoint, BetaFunction, CouplingSpace, RGFlow, FixedPointStability};
use crate::wick::WickProduct;
use crate::holders::holder_exponent;

/// Universal learning class for agent dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UniversalClass {
    /// KPZ: height function growth with competition.
    /// ∂ₜh = ν∇²h + λ(∇h)² + ξ
    KPZ,
    /// Allen-Cahn: bistable phase transitions.
    /// ∂ₜφ = Δφ + φ - φ³ + ξ
    AllenCahn,
    /// Φ⁴: critical point with scaling behavior.
    /// ∂ₜφ = Δφ - m²φ - λ:φ³: + ξ
    Phi4,
    /// Linear diffusion: trivial (Gaussian) fixed point.
    /// ∂ₜu = Δu - m²u + ξ
    Diffusive,
    /// Mean-field: all agents see the same average.
    MeanField,
}

impl std::fmt::Display for UniversalClass {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            UniversalClass::KPZ => write!(f, "KPZ (Growth)"),
            UniversalClass::AllenCahn => write!(f, "Allen-Cahn (Phase Transition)"),
            UniversalClass::Phi4 => write!(f, "Φ⁴ (Critical)"),
            UniversalClass::Diffusive => write!(f, "Diffusive (Gaussian)"),
            UniversalClass::MeanField => write!(f, "Mean-Field"),
        }
    }
}

/// KPZ equation for agent learning.
///
/// Models competitive learning where agents compete for resources.
/// The "height" h(x,t) represents accumulated knowledge/advantage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KPZDynamics {
    /// Diffusion coefficient ν.
    pub viscosity: f64,
    /// Nonlinearity strength λ.
    pub coupling: f64,
    /// Noise intensity.
    pub noise: f64,
    /// Spatial dimension.
    pub spatial_dim: usize,
}

impl KPZDynamics {
    pub fn new(viscosity: f64, coupling: f64, noise: f64, spatial_dim: usize) -> Self {
        Self { viscosity, coupling, noise, spatial_dim }
    }

    /// Classify singularity.
    pub fn singularity(&self) -> SingularityClass {
        // KPZ is singular in d ≥ 1 because (∇h)² is ill-defined
        if self.spatial_dim >= 1 {
            SingularityClass::Singular
        } else {
            SingularityClass::Regular
        }
    }

    /// The SPDE representation.
    pub fn to_spde(&self, grid_size: usize) -> AgentSPDE {
        AgentSPDE {
            dimension: grid_size,
            spatial_dim: self.spatial_dim,
            diffusion: self.viscosity * DMatrix::identity(grid_size, grid_size),
            nonlinearity: NonlinearityType::GradientSquared,
            noise_intensity: self.noise,
            mass: 0.0,
            coupling: self.coupling,
        }
    }

    /// KPZ scaling exponent: height fluctuations scale as t^{1/3} in 1d.
    pub fn scaling_exponent(&self) -> f64 {
        match self.spatial_dim {
            1 => 1.0 / 3.0,
            2 => 0.24,  // approximate
            _ => 0.0,   // roughening transition at d=2
        }
    }

    /// The Tracy-Widom distribution describes KPZ fluctuations in 1d.
    /// This returns a parameterization of the distribution shape.
    pub fn fluctuation_distribution(&self) -> (f64, f64) {
        // (mean, variance) of Tracy-Widom GUE
        (-1.7711, 0.8132)
    }

    /// Simulate KPZ dynamics and return height field trajectory.
    pub fn simulate(&self, grid_size: usize, domain_length: f64, dt: f64, steps: usize) -> Vec<DVector<f64>> {
        let spde = self.to_spde(grid_size);
        let initial = DVector::zeros(grid_size);
        solve_spde(&spde, &initial, grid_size, domain_length, dt, steps)
    }

    /// Compute the associated fixed point.
    pub fn fixed_point(&self) -> FixedPoint {
        FixedPoint::kpz(self.spatial_dim)
    }
}

/// Allen-Cahn equation for agent learning.
///
/// Models bistable learning dynamics: agents can be in one of two
/// stable states (e.g., "learned" vs "not learned").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllenCahnDynamics {
    /// Diffusion coefficient.
    pub diffusion: f64,
    /// Double-well potential strength.
    pub coupling: f64,
    /// Noise intensity.
    pub noise: f64,
    /// Spatial dimension.
    pub spatial_dim: usize,
}

impl AllenCahnDynamics {
    pub fn new(diffusion: f64, coupling: f64, noise: f64, spatial_dim: usize) -> Self {
        Self { diffusion, coupling, noise, spatial_dim }
    }

    /// Classify singularity.
    pub fn singularity(&self) -> SingularityClass {
        // Allen-Cahn has φ³ nonlinearity
        if self.spatial_dim >= 2 {
            SingularityClass::Singular
        } else if self.spatial_dim == 1 {
            SingularityClass::Singular // φ³·ξ still singular in 1d
        } else {
            SingularityClass::Regular
        }
    }

    /// The double-well potential V(φ) = (φ²-1)² / 4.
    pub fn potential(&self, phi: f64) -> f64 {
        (phi * phi - 1.0).powi(2) / 4.0
    }

    /// Gradient of potential: V'(φ) = φ³ - φ.
    pub fn potential_gradient(&self, phi: f64) -> f64 {
        phi * phi * phi - phi
    }

    /// The two minima of the double-well.
    pub fn minima(&self) -> (f64, f64) {
        (-1.0, 1.0)
    }

    /// Interface energy between phases.
    pub fn interface_energy(&self, state: &DVector<f64>, dx: f64) -> f64 {
        energy_functional(state, -1.0, self.coupling, dx)
    }

    /// Simulate Allen-Cahn dynamics.
    pub fn simulate(&self, grid_size: usize, domain_length: f64, dt: f64, steps: usize) -> Vec<DVector<f64>> {
        let spde = AgentSPDE {
            dimension: grid_size,
            spatial_dim: self.spatial_dim,
            diffusion: self.diffusion * DMatrix::identity(grid_size, grid_size),
            nonlinearity: NonlinearityType::Cubic,
            noise_intensity: self.noise,
            mass: -1.0, // Negative mass for double-well
            coupling: self.coupling,
        };
        let initial = DVector::zeros(grid_size);
        solve_spde(&spde, &initial, grid_size, domain_length, dt, steps)
    }
}

/// Φ⁴ critical dynamics for agent learning.
///
/// At criticality (m² = m²_c), the system exhibits scale-invariant
/// correlations. Agent beliefs fluctuate at all scales simultaneously.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phi4Dynamics {
    /// Mass parameter m² (tune to criticality).
    pub mass: f64,
    /// Coupling constant λ.
    pub coupling: f64,
    /// Noise intensity.
    pub noise: f64,
    /// Spatial dimension.
    pub spatial_dim: usize,
    /// UV cutoff for renormalization.
    pub cutoff: f64,
}

impl Phi4Dynamics {
    pub fn new(mass: f64, coupling: f64, noise: f64, spatial_dim: usize) -> Self {
        Self { mass, coupling, noise, spatial_dim, cutoff: 100.0 }
    }

    /// Is the system at critical mass?
    pub fn is_critical(&self) -> bool {
        // Critical mass m²_c ≈ -λC where C is the Wick constant
        // Simplified: critical when m² ≈ some negative value
        let critical_mass = -self.coupling * 0.5; // approximate
        (self.mass - critical_mass).abs() < 0.1
    }

    /// The Wick-ordered version of the nonlinearity.
    pub fn wick_ordered_nonlinearity(&self, phi: &DVector<f64>, variance: f64) -> DVector<f64> {
        let wick = WickProduct::new(3);
        wick.evaluate_vector(phi, variance)
    }

    /// The correlation length ξ ~ 1/|m² - m²_c|^ν.
    pub fn correlation_length(&self) -> f64 {
        let m_c = -self.coupling * 0.5;
        let delta_m = (self.mass - m_c).abs();
        if delta_m < 1e-10 {
            f64::INFINITY
        } else {
            let nu = 1.0 / (4.0 - self.spatial_dim as f64).max(0.01);
            1.0 / delta_m.powf(nu)
        }
    }

    /// The associated RG fixed point.
    pub fn fixed_point(&self) -> FixedPoint {
        if self.spatial_dim < 4 {
            let epsilon = 4.0 - self.spatial_dim as f64;
            FixedPoint::wilson_fisher(epsilon)
        } else {
            FixedPoint::gaussian(self.spatial_dim)
        }
    }

    /// The β-function for this theory.
    pub fn beta_function(&self) -> BetaFunction {
        BetaFunction::Phi4SuperRenorm {
            dim: self.spatial_dim,
            coefficient: 3.0 / (16.0 * std::f64::consts::PI.powi(2)),
        }
    }

    /// Simulate Φ⁴ dynamics.
    pub fn simulate(&self, grid_size: usize, domain_length: f64, dt: f64, steps: usize) -> Vec<DVector<f64>> {
        let spde = AgentSPDE {
            dimension: grid_size,
            spatial_dim: self.spatial_dim,
            diffusion: DMatrix::identity(grid_size, grid_size),
            nonlinearity: NonlinearityType::Cubic,
            noise_intensity: self.noise,
            mass: self.mass,
            coupling: self.coupling,
        };
        let initial = DVector::zeros(grid_size);
        solve_spde(&spde, &initial, grid_size, domain_length, dt, steps)
    }
}

/// Classify an agent SPDE into its universal class.
pub fn classify_universal(spde: &AgentSPDE) -> UniversalClass {
    match spde.nonlinearity {
        NonlinearityType::Linear => UniversalClass::Diffusive,
        NonlinearityType::GradientSquared => UniversalClass::KPZ,
        NonlinearityType::Quadratic => {
            // Quadratic with gradient structure → KPZ
            // Quadratic without gradient → could be Φ⁴-like
            UniversalClass::KPZ
        }
        NonlinearityType::Cubic => {
            if spde.mass < 0.0 {
                // Negative mass → double-well → Allen-Cahn
                UniversalClass::AllenCahn
            } else {
                // Positive mass → Φ⁴ critical
                UniversalClass::Phi4
            }
        }
        NonlinearityType::Polynomial(deg) => {
            if deg <= 2 { UniversalClass::KPZ }
            else { UniversalClass::Phi4 }
        }
    }
}

/// Get the characteristic exponents for a universal class.
pub fn critical_exponents(class: UniversalClass, spatial_dim: usize) -> CriticalExponents {
    match class {
        UniversalClass::KPZ => CriticalExponents {
            dynamic: 1.5,       // z = 3/2 in 1d KPZ
            roughness: 0.5,     // α = 1/2 in 1d KPZ
            correlation: 1.0,   // ν = 1
            class_name: class,
            spatial_dim,
        },
        UniversalClass::AllenCahn => CriticalExponents {
            dynamic: 2.0,       // z = 2 (diffusive)
            roughness: 0.0,     // Sharp interfaces
            correlation: 1.0 / (4.0 - spatial_dim as f64).max(0.01),
            class_name: class,
            spatial_dim,
        },
        UniversalClass::Phi4 => {
            let epsilon = (4.0 - spatial_dim as f64).max(0.0);
            CriticalExponents {
                dynamic: 2.0 - epsilon * 0.026, // η correction
                roughness: (2.0 - epsilon) * 0.5,
                correlation: 1.0 / epsilon.max(0.01),
                class_name: class,
                spatial_dim,
            }
        }
        UniversalClass::Diffusive => CriticalExponents {
            dynamic: 2.0,
            roughness: 1.0,
            correlation: 0.5,
            class_name: class,
            spatial_dim,
        },
        UniversalClass::MeanField => CriticalExponents {
            dynamic: 2.0,
            roughness: 1.0,
            correlation: 0.5,
            class_name: class,
            spatial_dim,
        },
    }
}

/// Critical exponents for a universal class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalExponents {
    /// Dynamic exponent z: time scales as length^z.
    pub dynamic: f64,
    /// Roughness exponent α: fluctuations scale as length^α.
    pub roughness: f64,
    /// Correlation length exponent ν.
    pub correlation: f64,
    /// The universal class.
    pub class_name: UniversalClass,
    /// Spatial dimension.
    pub spatial_dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_universal_class_display() {
        assert!(format!("{}", UniversalClass::KPZ).contains("KPZ"));
        assert!(format!("{}", UniversalClass::AllenCahn).contains("Phase"));
        assert!(format!("{}", UniversalClass::Phi4).contains("Critical"));
    }

    #[test]
    fn test_kpz_singularity() {
        let kpz = KPZDynamics::new(1.0, 1.0, 1.0, 1);
        assert_eq!(kpz.singularity(), SingularityClass::Singular);
    }

    #[test]
    fn test_kpz_scaling_exponent_1d() {
        let kpz = KPZDynamics::new(1.0, 1.0, 1.0, 1);
        assert_relative_eq!(kpz.scaling_exponent(), 1.0 / 3.0);
    }

    #[test]
    fn test_kpz_to_spde() {
        let kpz = KPZDynamics::new(1.0, 0.5, 0.1, 1);
        let spde = kpz.to_spde(10);
        assert_eq!(spde.nonlinearity, NonlinearityType::GradientSquared);
        assert_relative_eq!(spde.noise_intensity, 0.1);
    }

    #[test]
    fn test_kpz_fluctuation_distribution() {
        let kpz = KPZDynamics::new(1.0, 1.0, 1.0, 1);
        let (mean, var) = kpz.fluctuation_distribution();
        assert!(mean < 0.0);
        assert!(var > 0.0);
    }

    #[test]
    fn test_kpz_simulate() {
        let kpz = KPZDynamics::new(1.0, 0.1, 0.1, 1);
        let traj = kpz.simulate(20, 1.0, 0.001, 10);
        assert_eq!(traj.len(), 11);
        assert_eq!(traj[0].len(), 20);
    }

    #[test]
    fn test_allen_cahn_singularity() {
        let ac = AllenCahnDynamics::new(1.0, 1.0, 0.1, 2);
        assert_eq!(ac.singularity(), SingularityClass::Singular);
    }

    #[test]
    fn test_allen_cahn_potential() {
        let ac = AllenCahnDynamics::new(1.0, 1.0, 0.1, 1);
        assert_relative_eq!(ac.potential(0.0), 0.25);
        assert_relative_eq!(ac.potential(1.0), 0.0);
        assert_relative_eq!(ac.potential(-1.0), 0.0);
    }

    #[test]
    fn test_allen_cahn_minima() {
        let ac = AllenCahnDynamics::new(1.0, 1.0, 0.1, 1);
        let (m1, m2) = ac.minima();
        assert_relative_eq!(m1, -1.0);
        assert_relative_eq!(m2, 1.0);
    }

    #[test]
    fn test_allen_cahn_gradient() {
        let ac = AllenCahnDynamics::new(1.0, 1.0, 0.1, 1);
        // V'(φ) = φ³ - φ = 0 at φ = 0, ±1
        assert_relative_eq!(ac.potential_gradient(0.0), 0.0);
        assert_relative_eq!(ac.potential_gradient(1.0), 0.0);
        assert_relative_eq!(ac.potential_gradient(-1.0), 0.0);
    }

    #[test]
    fn test_phi4_criticality() {
        let phi4 = Phi4Dynamics::new(-0.5, 1.0, 0.1, 1);
        // m_c ≈ -λ * 0.5 = -0.5, so this should be near critical
        assert!(phi4.is_critical());
    }

    #[test]
    fn test_phi4_not_critical() {
        let phi4 = Phi4Dynamics::new(5.0, 1.0, 0.1, 1);
        assert!(!phi4.is_critical());
    }

    #[test]
    fn test_phi4_correlation_length() {
        let phi4 = Phi4Dynamics::new(5.0, 1.0, 0.1, 1);
        let xi = phi4.correlation_length();
        assert!(xi.is_finite());
        assert!(xi > 0.0);
    }

    #[test]
    fn test_phi4_correlation_length_critical() {
        let phi4 = Phi4Dynamics::new(-0.5, 1.0, 0.1, 1);
        let xi = phi4.correlation_length();
        // At criticality, correlation length diverges
        assert!(xi > 10.0 || xi.is_infinite());
    }

    #[test]
    fn test_phi4_fixed_point() {
        let phi4 = Phi4Dynamics::new(1.0, 1.0, 0.1, 3);
        let fp = phi4.fixed_point();
        assert!(fp.couplings.coupling > 0.0); // Wilson-Fisher
    }

    #[test]
    fn test_classify_universal_linear() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Linear,
            noise_intensity: 1.0, mass: 1.0, coupling: 0.0,
        };
        assert_eq!(classify_universal(&spde), UniversalClass::Diffusive);
    }

    #[test]
    fn test_classify_universal_kpz() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::GradientSquared,
            noise_intensity: 1.0, mass: 1.0, coupling: 1.0,
        };
        assert_eq!(classify_universal(&spde), UniversalClass::KPZ);
    }

    #[test]
    fn test_classify_universal_allen_cahn() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Cubic,
            noise_intensity: 1.0, mass: -1.0, coupling: 1.0,
        };
        assert_eq!(classify_universal(&spde), UniversalClass::AllenCahn);
    }

    #[test]
    fn test_classify_universal_phi4() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Cubic,
            noise_intensity: 1.0, mass: 1.0, coupling: 1.0,
        };
        assert_eq!(classify_universal(&spde), UniversalClass::Phi4);
    }

    #[test]
    fn test_critical_exponents_kpz() {
        let ce = critical_exponents(UniversalClass::KPZ, 1);
        assert_relative_eq!(ce.dynamic, 1.5);
    }

    #[test]
    fn test_critical_exponents_phi4() {
        let ce = critical_exponents(UniversalClass::Phi4, 3);
        assert!(ce.correlation > 0.0);
    }

    #[test]
    fn test_kpz_fixed_point() {
        let kpz = KPZDynamics::new(1.0, 1.0, 1.0, 1);
        let fp = kpz.fixed_point();
        assert_eq!(fp.name, "KPZ");
    }
}
