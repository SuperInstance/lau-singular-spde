//! Stochastic PDE formulation of agent learning dynamics.
//!
//! Models agent belief evolution as solutions to SPDEs of the form:
//!   ∂ₜu = Lu + F(u) + ξ
//! where L is a linear differential operator, F is a nonlinearity,
//! and ξ is space-time white noise.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use std::fmt;

/// Classification of SPDE singularity type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SingularityClass {
    /// Subcritical: solution is a classical function, no renormalization needed.
    Regular,
    /// Critical: solution barely fails to be classical, logarithmic divergences.
    Critical,
    /// Supercritical: solution is a distribution, renormalization required.
    Singular,
}

impl fmt::Display for SingularityClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SingularityClass::Regular => write!(f, "Regular"),
            SingularityClass::Critical => write!(f, "Critical"),
            SingularityClass::Singular => write!(f, "Singular"),
        }
    }
}

/// Space-time white noise realization on a grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteNoise {
    /// Number of spatial grid points.
    pub grid_size: usize,
    /// Spatial domain [0, L].
    pub domain_length: f64,
    /// Noise variance (intensity).
    pub intensity: f64,
    /// Realization values at grid points for a single time step.
    pub values: DVector<f64>,
}

impl WhiteNoise {
    /// Generate a new white noise realization with given parameters.
    pub fn new(grid_size: usize, domain_length: f64, intensity: f64) -> Self {
        let mut rng = rand::thread_rng();
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, intensity).unwrap();
        let vals: Vec<f64> = (0..grid_size).map(|_| normal.sample(&mut rng)).collect();
        Self {
            grid_size,
            domain_length,
            intensity,
            values: DVector::from_vec(vals),
        }
    }

    /// Create a deterministic white noise from given values.
    pub fn from_values(grid_size: usize, domain_length: f64, intensity: f64, values: Vec<f64>) -> Self {
        Self {
            grid_size,
            domain_length,
            intensity,
            values: DVector::from_vec(values),
        }
    }

    /// Spatial step size.
    pub fn dx(&self) -> f64 {
        self.domain_length / self.grid_size as f64
    }

    /// Covariance operator: white noise has identity covariance.
    pub fn covariance(&self) -> DMatrix<f64> {
        DMatrix::identity(self.grid_size, self.grid_size)
    }
}

/// A stochastic PDE representing agent learning dynamics.
///
/// General form: ∂ₜu = Lu + F(u, ∇u, ...) + σξ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSPDE {
    /// Dimension of the state space (number of agent beliefs).
    pub dimension: usize,
    /// Spatial dimension of the domain.
    pub spatial_dim: usize,
    /// Linear operator coefficients (diffusion matrix).
    pub diffusion: DMatrix<f64>,
    /// Nonlinearity type.
    pub nonlinearity: NonlinearityType,
    /// Noise intensity σ.
    pub noise_intensity: f64,
    /// Mass parameter m².
    pub mass: f64,
    /// Coupling constant λ for nonlinear terms.
    pub coupling: f64,
}

/// Types of nonlinearities in agent learning SPDEs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NonlinearityType {
    /// Linear: F(u) = 0
    Linear,
    /// Quadratic: F(u) = λu² (KPZ-like growth)
    Quadratic,
    /// Cubic: F(u) = -λu³ + u (Allen-Cahn / Φ⁴)
    Cubic,
    /// Gradient squared: F(u) = λ(∇u)² (KPZ)
    GradientSquared,
    /// General polynomial with given degree
    Polynomial(u32),
}

/// Classify the singularity of an agent SPDE based on its parameters.
///
/// An SPDE is singular if the product F(u)ξ is ill-defined in the
/// distributional sense. This happens when the regularity exponent
/// of the driving noise minus the scaling dimension is too negative.
pub fn classify_singularity(spde: &AgentSPDE, spatial_dim: usize) -> SingularityClass {
    // The key criterion: in d spatial dimensions, space-time white noise
    // has regularity α = -(d/2 + 1) - ε for any ε > 0.
    // The nonlinearity F(u) requires products of distributions.
    // If 2α + scaling_dimension < 0, the product is ill-defined → singular.
    let noise_regularity = -(spatial_dim as f64 / 2.0 + 1.0);

    match spde.nonlinearity {
        NonlinearityType::Linear => SingularityClass::Regular,
        NonlinearityType::Quadratic | NonlinearityType::GradientSquared => {
            // Need u·ξ product. u has regularity α + 2 (from heat kernel smoothing).
            let u_regularity = noise_regularity + 2.0;
            let product_regularity = u_regularity + noise_regularity;
            if product_regularity < -(spatial_dim as f64) / 2.0 {
                SingularityClass::Singular
            } else if product_regularity.abs() < 0.01 {
                SingularityClass::Critical
            } else {
                SingularityClass::Regular
            }
        }
        NonlinearityType::Cubic => {
            // Need u³ · ξ. Even more singular.
            let u_regularity = noise_regularity + 2.0;
            let product_regularity = 3.0 * u_regularity + noise_regularity;
            if product_regularity <= 0.0 {
                SingularityClass::Singular
            } else if product_regularity < 0.01 {
                SingularityClass::Critical
            } else {
                SingularityClass::Regular
            }
        }
        NonlinearityType::Polynomial(deg) => {
            let u_regularity = noise_regularity + 2.0;
            let product_regularity = (deg as f64) * u_regularity + noise_regularity;
            if deg >= 3 || product_regularity < -(spatial_dim as f64) / 2.0 {
                SingularityClass::Singular
            } else if product_regularity.abs() < 0.01 {
                SingularityClass::Critical
            } else {
                SingularityClass::Regular
            }
        }
    }
}

/// Build the discrete Laplacian operator on a grid of size n.
pub fn discrete_laplacian(n: usize, dx: f64) -> DMatrix<f64> {
    let mut mat = DMatrix::zeros(n, n);
    let coeff = 1.0 / (dx * dx);
    for i in 0..n {
        mat[(i, i)] = -2.0 * coeff;
        if i > 0 {
            mat[(i, i - 1)] = coeff;
        }
        if i < n - 1 {
            mat[(i, i + 1)] = coeff;
        }
    }
    // Periodic boundary
    if n > 1 {
        mat[(0, n - 1)] = coeff;
        mat[(n - 1, 0)] = coeff;
    }
    mat
}

/// Euler-Maruyama step for an SPDE.
pub fn euler_maruyama_step(
    state: &DVector<f64>,
    laplacian: &DMatrix<f64>,
    spde: &AgentSPDE,
    noise: &WhiteNoise,
    dt: f64,
) -> DVector<f64> {
    // ∂ₜu = Δu - m²u + F(u) + σξ
    let linear = laplacian * state - spde.mass * state;
    let nonlinear = apply_nonlinearity(state, spde.nonlinearity, spde.coupling);
    let stochastic = spde.noise_intensity * &noise.values * dt.sqrt();

    state + (linear + nonlinear) * dt + stochastic
}

/// Apply the nonlinear term F(u) to a state vector.
pub fn apply_nonlinearity(
    state: &DVector<f64>,
    nl_type: NonlinearityType,
    coupling: f64,
) -> DVector<f64> {
    match nl_type {
        NonlinearityType::Linear => DVector::zeros(state.len()),
        NonlinearityType::Quadratic => {
            let mut result = state.clone();
            result.apply(|x| *x = coupling * *x * *x);
            result
        }
        NonlinearityType::Cubic => {
            // Allen-Cahn: u - λu³
            let mut result = state.clone();
            result.apply(|x| *x = *x - coupling * *x * *x * *x);
            result
        }
        NonlinearityType::GradientSquared => {
            // Approximate (∇u)² using finite differences
            let n = state.len();
            let mut result = DVector::zeros(n);
            for i in 0..n {
                let ip = (i + 1) % n;
                let im = if i > 0 { i - 1 } else { n - 1 };
                let grad = (state[ip] - state[im]) / 2.0;
                result[i] = coupling * grad * grad;
            }
            result
        }
        NonlinearityType::Polynomial(deg) => {
            let mut result = state.clone();
            result.apply(|x| {
                *x = coupling * x.powi(deg as i32);
            });
            result
        }
    }
}

/// Solve an SPDE over multiple time steps, returning trajectory.
pub fn solve_spde(
    spde: &AgentSPDE,
    initial: &DVector<f64>,
    grid_size: usize,
    domain_length: f64,
    dt: f64,
    steps: usize,
) -> Vec<DVector<f64>> {
    let dx = domain_length / grid_size as f64;
    let laplacian = discrete_laplacian(grid_size, dx);
    let mut trajectory = vec![initial.clone()];
    let mut state = initial.clone();

    for _ in 0..steps {
        let noise = WhiteNoise::new(grid_size, domain_length, 1.0);
        state = euler_maruyama_step(&state, &laplacian, spde, &noise, dt);
        trajectory.push(state.clone());
    }

    trajectory
}

/// Compute the energy functional E[u] for an Allen-Cahn type SPDE.
/// E[u] = ∫ (½|∇u|² + V(u)) dx where V(u) = ½m²u² + ¼λu⁴
pub fn energy_functional(
    state: &DVector<f64>,
    mass: f64,
    coupling: f64,
    dx: f64,
) -> f64 {
    let n = state.len();
    let mut gradient_energy = 0.0;
    for i in 0..n {
        let ip = (i + 1) % n;
        let grad = (state[ip] - state[i]) / dx;
        gradient_energy += grad * grad;
    }
    gradient_energy *= 0.5 * dx;

    let mut potential_energy = 0.0;
    for i in 0..n {
        let u = state[i];
        potential_energy += 0.5 * mass * u * u + 0.25 * coupling * u * u * u * u;
    }
    potential_energy *= dx;

    gradient_energy + potential_energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_white_noise_creation() {
        let wn = WhiteNoise::new(100, 1.0, 1.0);
        assert_eq!(wn.grid_size, 100);
        assert_eq!(wn.values.len(), 100);
    }

    #[test]
    fn test_white_noise_dx() {
        let wn = WhiteNoise::new(50, 2.0, 1.0);
        assert_relative_eq!(wn.dx(), 0.04);
    }

    #[test]
    fn test_white_noise_covariance() {
        let wn = WhiteNoise::new(10, 1.0, 1.0);
        let cov = wn.covariance();
        assert_eq!(cov.nrows(), 10);
        assert_eq!(cov.ncols(), 10);
        // Identity
        for i in 0..10 {
            assert_relative_eq!(cov[(i, i)], 1.0);
        }
    }

    #[test]
    fn test_singularity_classification_linear() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Linear,
            noise_intensity: 1.0, mass: 1.0, coupling: 0.0,
        };
        assert_eq!(classify_singularity(&spde, 1), SingularityClass::Regular);
    }

    #[test]
    fn test_singularity_classification_quadratic_1d() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Quadratic,
            noise_intensity: 1.0, mass: 1.0, coupling: 1.0,
        };
        // In 1d: noise_regularity = -1.5, u_regularity = 0.5
        // product = 0.5 + (-1.5) = -1.0, threshold = -0.5 → singular
        assert_eq!(classify_singularity(&spde, 1), SingularityClass::Singular);
    }

    #[test]
    fn test_singularity_classification_cubic_1d() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Cubic,
            noise_intensity: 1.0, mass: 1.0, coupling: 1.0,
        };
        assert_eq!(classify_singularity(&spde, 1), SingularityClass::Singular);
    }

    #[test]
    fn test_discrete_laplacian_size() {
        let n = 10;
        let dx = 0.1;
        let lap = discrete_laplacian(n, dx);
        assert_eq!(lap.nrows(), n);
        assert_eq!(lap.ncols(), n);
    }

    #[test]
    fn test_discrete_laplacian_constant_vector() {
        let n = 20;
        let dx = 0.1;
        let lap = discrete_laplacian(n, dx);
        let const_vec = DVector::from_element(n, 1.0);
        let result = &lap * &const_vec;
        // Laplacian of constant should be zero
        for i in 0..n {
            assert_relative_eq!(result[i], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_discrete_laplacian_linear() {
        let n = 100;
        let dx = 0.1;
        let lap = discrete_laplacian(n, dx);
        // u = x → Δu = 0
        let linear: Vec<f64> = (0..n).map(|i| i as f64 * dx).collect();
        let u = DVector::from_vec(linear);
        let result = &lap * &u;
        // Interior points should be ~0
        for i in 1..n - 1 {
            assert_relative_eq!(result[i], 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_apply_nonlinearity_linear() {
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = apply_nonlinearity(&state, NonlinearityType::Linear, 1.0);
        assert_eq!(result, DVector::zeros(3));
    }

    #[test]
    fn test_apply_nonlinearity_quadratic() {
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = apply_nonlinearity(&state, NonlinearityType::Quadratic, 2.0);
        assert_relative_eq!(result[0], 2.0);
        assert_relative_eq!(result[1], 8.0);
        assert_relative_eq!(result[2], 18.0);
    }

    #[test]
    fn test_apply_nonlinearity_cubic() {
        let state = DVector::from_vec(vec![2.0]);
        // u - λu³ = 2 - 1*8 = -6
        let result = apply_nonlinearity(&state, NonlinearityType::Cubic, 1.0);
        assert_relative_eq!(result[0], -6.0);
    }

    #[test]
    fn test_euler_maruyama_step_shape() {
        let n = 10;
        let spde = AgentSPDE {
            dimension: n, spatial_dim: 1,
            diffusion: DMatrix::identity(n, n),
            nonlinearity: NonlinearityType::Linear,
            noise_intensity: 0.1, mass: 1.0, coupling: 0.0,
        };
        let state = DVector::from_element(n, 0.5);
        let lap = discrete_laplacian(n, 0.1);
        let noise = WhiteNoise::new(n, 1.0, 1.0);
        let result = euler_maruyama_step(&state, &lap, &spde, &noise, 0.001);
        assert_eq!(result.len(), n);
    }

    #[test]
    fn test_solve_spde_trajectory_length() {
        let n = 10;
        let spde = AgentSPDE {
            dimension: n, spatial_dim: 1,
            diffusion: DMatrix::identity(n, n),
            nonlinearity: NonlinearityType::Linear,
            noise_intensity: 0.1, mass: 1.0, coupling: 0.0,
        };
        let initial = DVector::zeros(n);
        let traj = solve_spde(&spde, &initial, n, 1.0, 0.001, 50);
        assert_eq!(traj.len(), 51); // initial + 50 steps
    }

    #[test]
    fn test_energy_functional_zero_state() {
        let state = DVector::zeros(10);
        let e = energy_functional(&state, 1.0, 1.0, 0.1);
        assert_relative_eq!(e, 0.0);
    }

    #[test]
    fn test_energy_functional_positive() {
        let state = DVector::from_element(10, 1.0);
        let e = energy_functional(&state, 1.0, 1.0, 0.1);
        assert!(e > 0.0);
    }

    #[test]
    fn test_singularity_display() {
        assert_eq!(format!("{}", SingularityClass::Regular), "Regular");
        assert_eq!(format!("{}", SingularityClass::Singular), "Singular");
        assert_eq!(format!("{}", SingularityClass::Critical), "Critical");
    }

    #[test]
    fn test_white_noise_from_values() {
        let wn = WhiteNoise::from_values(3, 1.0, 1.0, vec![1.0, 2.0, 3.0]);
        assert_eq!(wn.values.len(), 3);
        assert_relative_eq!(wn.values[0], 1.0);
        assert_relative_eq!(wn.values[1], 2.0);
        assert_relative_eq!(wn.values[2], 3.0);
    }

    #[test]
    fn test_singularity_classification_polynomial() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Polynomial(4),
            noise_intensity: 1.0, mass: 1.0, coupling: 1.0,
        };
        assert_eq!(classify_singularity(&spde, 1), SingularityClass::Singular);
    }
}
