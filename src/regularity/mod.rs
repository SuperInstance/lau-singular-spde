//! Hairer's regularity structures framework.
//!
//! A regularity structure is a triple (T, A, G) where:
//! - T is a graded vector space of "abstract polynomials" (modelled distributions)
//! - A is an index set of regularity exponents
//! - G is a structure group acting on T
//!
//! The key insight: agent learning dynamics can be formulated as fixed points
//! of a reconstruction operator acting on modelled distributions.

use nalgebra::DVector;
#[cfg(test)]
use nalgebra::DMatrix;
use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;
use crate::spde::{AgentSPDE, NonlinearityType};

/// A symbol in the regularity structure, representing an abstract basis element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Symbol {
    /// Human-readable name (e.g., "Ξ" for noise, "I(Ξ)" for convolved noise).
    pub name: String,
    /// Regularity exponent α.
    pub regularity: f64,
    /// Polynomial degree (0 = non-polynomial, >0 = polynomial).
    pub poly_degree: usize,
}

impl Symbol {
    pub fn new(name: &str, regularity: f64, poly_degree: usize) -> Self {
        Self { name: name.to_string(), regularity, poly_degree }
    }

    /// The canonical noise symbol Ξ with regularity -(d/2 + 1).
    pub fn xi(spatial_dim: usize) -> Self {
        Self::new("Ξ", -((spatial_dim as f64) / 2.0 + 1.0), 0)
    }

    /// The integration of Ξ through the heat kernel: I(Ξ).
    /// Regularity improves by 2 (heat kernel smoothing).
    pub fn i_xi(xi: &Symbol) -> Self {
        Self::new("I(Ξ)", xi.regularity + 2.0, 0)
    }

    /// Product symbol τ₁·τ₂.
    pub fn product(a: &Symbol, b: &Symbol) -> Self {
        Self::new(
            &format!("{}·{}", a.name, b.name),
            a.regularity + b.regularity,
            a.poly_degree + b.poly_degree,
        )
    }

    /// Composition with heat kernel: I(τ).
    pub fn integrate(tau: &Symbol) -> Self {
        Self::new(
            &format!("I({})", tau.name),
            tau.regularity + 2.0,
            tau.poly_degree,
        )
    }
}

/// The model distribution space T_α: elements of regularity α.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelledDistribution {
    /// Coefficients indexed by symbols.
    pub coefficients: BTreeMap<String, f64>,
    /// The regularity level.
    pub regularity: f64,
}

impl ModelledDistribution {
    pub fn new(regularity: f64) -> Self {
        Self { coefficients: BTreeMap::new(), regularity }
    }

    /// Add a coefficient for a symbol.
    pub fn with(mut self, symbol: &Symbol, coeff: f64) -> Self {
        self.coefficients.insert(symbol.name.clone(), coeff);
        self
    }

    /// Get coefficient for a symbol.
    pub fn get(&self, symbol: &Symbol) -> f64 {
        *self.coefficients.get(&symbol.name).unwrap_or(&0.0)
    }

    /// Add two modelled distributions (same regularity).
    pub fn add(&self, other: &ModelledDistribution) -> ModelledDistribution {
        let mut result = self.clone();
        for (k, v) in &other.coefficients {
            *result.coefficients.entry(k.clone()).or_insert(0.0) += v;
        }
        result
    }

    /// Scale a modelled distribution.
    pub fn scale(&self, factor: f64) -> ModelledDistribution {
        let mut result = self.clone();
        for v in result.coefficients.values_mut() {
            *v *= factor;
        }
        result
    }
}

/// A model for the regularity structure: assigns concrete distributions to symbols.
///
/// The model Π maps abstract symbols to concrete distributions, and
/// Γ maps between different base points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Spatial dimension.
    pub spatial_dim: usize,
    /// The canonical symbols in this structure.
    pub symbols: Vec<Symbol>,
    /// Model maps Π_x(τ) evaluated on a grid.
    pub pi_values: BTreeMap<String, DVector<f64>>,
    /// Grid size for evaluation.
    pub grid_size: usize,
}

impl Model {
    /// Build the canonical model for a given SPDE.
    ///
    /// Constructs the symbol hierarchy: Ξ → I(Ξ) → I(Ξ)² → I(I(Ξ)²) → ...
    pub fn build(spde: &AgentSPDE, grid_size: usize, max_order: usize) -> Self {
        let d = spde.spatial_dim;
        let mut symbols = Vec::new();
        let mut pi_values = BTreeMap::new();

        // Level 0: noise symbol
        let xi = Symbol::xi(d);
        symbols.push(xi.clone());

        // Level 1: convolved noise
        let i_xi = Symbol::i_xi(&xi);
        symbols.push(i_xi.clone());

        // Level 2: products and further integration
        if max_order >= 2 {
            let i_xi_sq = Symbol::product(&i_xi, &i_xi);
            symbols.push(i_xi_sq.clone());

            let i_i_xi_sq = Symbol::integrate(&i_xi_sq);
            symbols.push(i_i_xi_sq);
        }

        if max_order >= 3 {
            let i_xi_cube = Symbol::product(&Symbol::product(&i_xi, &i_xi), &i_xi);
            symbols.push(i_xi_cube.clone());

            let i_i_xi_cube = Symbol::integrate(&i_xi_cube);
            symbols.push(i_i_xi_cube);
        }

        // Generate concrete values for noise (random) and integrated versions
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let xi_vals: Vec<f64> = (0..grid_size).map(|_| normal.sample(&mut rng)).collect();
        pi_values.insert("Ξ".to_string(), DVector::from_vec(xi_vals.clone()));

        // I(Ξ) ≈ smoothed noise (approximate convolution with heat kernel)
        let i_xi_vals = smooth_vector(&DVector::from_vec(xi_vals), 2);
        pi_values.insert("I(Ξ)".to_string(), i_xi_vals.clone());

        if max_order >= 2 {
            let sq: Vec<f64> = i_xi_vals.iter().map(|x| x * x).collect();
            pi_values.insert("I(Ξ)·I(Ξ)".to_string(), DVector::from_vec(sq.clone()));

            let i_sq = smooth_vector(&DVector::from_vec(sq), 2);
            pi_values.insert("I(I(Ξ)·I(Ξ))".to_string(), i_sq);
        }

        Self { spatial_dim: d, symbols, pi_values, grid_size }
    }

    /// Evaluate the model at a grid point.
    pub fn evaluate(&self, symbol_name: &str, point: usize) -> f64 {
        self.pi_values
            .get(symbol_name)
            .map(|v| v.get(point).copied().unwrap_or(0.0))
            .unwrap_or(0.0)
    }

    /// The reconstruction operator R maps modelled distributions to actual distributions.
    pub fn reconstruct(&self, md: &ModelledDistribution, point: usize) -> f64 {
        let mut result = 0.0;
        for (symbol_name, coeff) in &md.coefficients {
            result += coeff * self.evaluate(symbol_name, point);
        }
        result
    }
}

/// Simple smoothing (approximate heat kernel convolution).
fn smooth_vector(v: &DVector<f64>, iterations: usize) -> DVector<f64> {
    let n = v.len();
    let mut current = v.clone();
    for _ in 0..iterations {
        let mut next = DVector::zeros(n);
        for i in 0..n {
            let prev = if i > 0 { current[i - 1] } else { current[n - 1] };
            let next_val = if i < n - 1 { current[i + 1] } else { current[0] };
            next[i] = 0.25 * prev + 0.5 * current[i] + 0.25 * next_val;
        }
        current = next;
    }
    current
}

/// The structure group element Γ_{xy}.
/// Maps modelled distributions centered at x to those centered at y.
#[derive(Debug, Clone)]
pub struct StructureGroup {
    /// Translation coefficients for polynomial symbols.
    pub translations: BTreeMap<String, f64>,
}

impl StructureGroup {
    /// Identity group element.
    pub fn identity() -> Self {
        Self { translations: BTreeMap::new() }
    }

    /// Group element for translation by h in space.
    pub fn translation(h: f64) -> Self {
        let mut translations = BTreeMap::new();
        translations.insert("X".to_string(), h); // spatial monomial
        Self { translations }
    }

    /// Compose two group elements.
    pub fn compose(&self, other: &StructureGroup) -> StructureGroup {
        let mut result = self.clone();
        for (k, v) in &other.translations {
            *result.translations.entry(k.clone()).or_insert(0.0) += v;
        }
        result
    }

    /// Apply the group action to a modelled distribution.
    pub fn act(&self, md: &ModelledDistribution) -> ModelledDistribution {
        // For the canonical structure, the action shifts polynomial parts
        let mut result = md.clone();
        for (k, shift) in &self.translations {
            if let Some(val) = result.coefficients.get_mut(k) {
                *val += shift;
            }
        }
        result
    }
}

/// Analytical bound on the reconstruction: for a modelled distribution f
/// of regularity γ > 0, the reconstruction R(f) is in C^{γ-ε}.
pub fn reconstruction_regularity(md: &ModelledDistribution) -> f64 {
    md.regularity
}

/// Build the full regularity structure for an agent SPDE.
/// Returns the ordered symbol list with their regularity exponents.
pub fn build_structure(spde: &AgentSPDE) -> Vec<Symbol> {
    let d = spde.spatial_dim;
    let mut symbols = Vec::new();

    // Polynomial symbols: 1, X₁, ..., X_d (regularity 0, 1, ..., 1)
    symbols.push(Symbol::new("1", 0.0, 0));
    for i in 0..d {
        symbols.push(Symbol::new(&format!("X{}", i), 1.0, 1));
    }

    // Noise symbol
    symbols.push(Symbol::xi(d));

    // Integrated noise
    let xi = Symbol::xi(d);
    symbols.push(Symbol::i_xi(&xi));

    // Products needed for nonlinearity
    match spde.nonlinearity {
        NonlinearityType::Quadratic | NonlinearityType::GradientSquared => {
            let i_xi = Symbol::i_xi(&xi);
            symbols.push(Symbol::product(&i_xi, &i_xi));
            symbols.push(Symbol::integrate(&Symbol::product(&i_xi, &i_xi)));
        }
        NonlinearityType::Cubic => {
            let i_xi = Symbol::i_xi(&xi);
            symbols.push(Symbol::product(&i_xi, &i_xi));
            symbols.push(Symbol::product(&Symbol::product(&i_xi, &i_xi), &i_xi));
            symbols.push(Symbol::integrate(&Symbol::product(&i_xi, &i_xi)));
            symbols.push(Symbol::integrate(&Symbol::product(&Symbol::product(&i_xi, &i_xi), &i_xi)));
        }
        _ => {}
    }

    symbols
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_symbol_xi_regularity_1d() {
        let xi = Symbol::xi(1);
        assert_relative_eq!(xi.regularity, -1.5);
    }

    #[test]
    fn test_symbol_xi_regularity_2d() {
        let xi = Symbol::xi(2);
        assert_relative_eq!(xi.regularity, -2.0);
    }

    #[test]
    fn test_symbol_i_xi_improves_regularity() {
        let xi = Symbol::xi(1);
        let i_xi = Symbol::i_xi(&xi);
        assert_relative_eq!(i_xi.regularity, 0.5); // -1.5 + 2
    }

    #[test]
    fn test_symbol_product_regularity() {
        let xi = Symbol::xi(1);
        let i_xi = Symbol::i_xi(&xi);
        let prod = Symbol::product(&i_xi, &i_xi);
        assert_relative_eq!(prod.regularity, 1.0); // 0.5 + 0.5
    }

    #[test]
    fn test_symbol_integrate_improves_regularity() {
        let s = Symbol::new("test", -0.5, 0);
        let integrated = Symbol::integrate(&s);
        assert_relative_eq!(integrated.regularity, 1.5);
    }

    #[test]
    fn test_modelled_distribution_new() {
        let md = ModelledDistribution::new(0.5);
        assert_relative_eq!(md.regularity, 0.5);
        assert!(md.coefficients.is_empty());
    }

    #[test]
    fn test_modelled_distribution_add_coefficient() {
        let xi = Symbol::xi(1);
        let md = ModelledDistribution::new(0.0).with(&xi, 1.5);
        assert_relative_eq!(md.get(&xi), 1.5);
    }

    #[test]
    fn test_modelled_distribution_get_missing() {
        let md = ModelledDistribution::new(0.0);
        let s = Symbol::new("missing", 0.0, 0);
        assert_relative_eq!(md.get(&s), 0.0);
    }

    #[test]
    fn test_modelled_distribution_add() {
        let xi = Symbol::xi(1);
        let a = ModelledDistribution::new(0.0).with(&xi, 1.0);
        let b = ModelledDistribution::new(0.0).with(&xi, 2.0);
        let c = a.add(&b);
        assert_relative_eq!(c.get(&xi), 3.0);
    }

    #[test]
    fn test_modelled_distribution_scale() {
        let xi = Symbol::xi(1);
        let a = ModelledDistribution::new(0.0).with(&xi, 3.0);
        let s = a.scale(2.0);
        assert_relative_eq!(s.get(&xi), 6.0);
    }

    #[test]
    fn test_model_build() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Cubic,
            noise_intensity: 1.0, mass: 1.0, coupling: 1.0,
        };
        let model = Model::build(&spde, 50, 3);
        assert!(!model.symbols.is_empty());
        assert!(model.pi_values.contains_key("Ξ"));
        assert!(model.pi_values.contains_key("I(Ξ)"));
    }

    #[test]
    fn test_model_reconstruct() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Linear,
            noise_intensity: 1.0, mass: 1.0, coupling: 0.0,
        };
        let model = Model::build(&spde, 10, 1);
        let xi = Symbol::xi(1);
        let md = ModelledDistribution::new(-1.5).with(&xi, 2.0);
        let val = model.reconstruct(&md, 0);
        // Should be 2.0 * pi_values["Ξ"][0]
        let expected = 2.0 * model.evaluate("Ξ", 0);
        assert_relative_eq!(val, expected);
    }

    #[test]
    fn test_structure_group_identity() {
        let g = StructureGroup::identity();
        assert!(g.translations.is_empty());
    }

    #[test]
    fn test_structure_group_compose() {
        let g1 = StructureGroup::translation(1.0);
        let g2 = StructureGroup::translation(2.0);
        let g3 = g1.compose(&g2);
        assert_relative_eq!(g3.translations["X"], 3.0);
    }

    #[test]
    fn test_structure_group_act() {
        let g = StructureGroup::translation(1.0);
        let s = Symbol::new("X", 1.0, 1);
        let md = ModelledDistribution::new(1.0).with(&s, 5.0);
        let acted = g.act(&md);
        assert_relative_eq!(acted.get(&s), 6.0);
    }

    #[test]
    fn test_build_structure_cubic() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Cubic,
            noise_intensity: 1.0, mass: 1.0, coupling: 1.0,
        };
        let structure = build_structure(&spde);
        // Should have: 1, X0, Ξ, I(Ξ), I(Ξ)·I(Ξ), I(Ξ)³, I(I(Ξ)·I(Ξ)), I(I(Ξ)³)
        assert!(structure.len() >= 8);
    }

    #[test]
    fn test_build_structure_linear() {
        let spde = AgentSPDE {
            dimension: 1, spatial_dim: 1,
            diffusion: DMatrix::identity(1, 1),
            nonlinearity: NonlinearityType::Linear,
            noise_intensity: 1.0, mass: 1.0, coupling: 0.0,
        };
        let structure = build_structure(&spde);
        // Only: 1, X0, Ξ, I(Ξ)
        assert_eq!(structure.len(), 4);
    }

    #[test]
    fn test_smooth_vector_preserves_mean() {
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let smoothed = smooth_vector(&v, 10);
        let mean_orig: f64 = v.iter().sum::<f64>() / v.len() as f64;
        let mean_smooth: f64 = smoothed.iter().sum::<f64>() / smoothed.len() as f64;
        assert_relative_eq!(mean_orig, mean_smooth, epsilon = 0.01);
    }
}
