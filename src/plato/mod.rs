//! PLATO agent classification: which agents need renormalized updates?
//!
//! PLATO agents have different learning rules. Some are "regular" (classical
//! solutions exist) and some are "singular" (need renormalization via
//! Wick ordering and regularity structures).
//!
//! This module maps PLATO agent types to their SPDE classification and
//! determines which ones need the full renormalization machinery.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::spde::{
    AgentSPDE, NonlinearityType, SingularityClass,
    classify_singularity, WhiteNoise, discrete_laplacian, euler_maruyama_step,
};
use crate::regularity::{Model, ModelledDistribution, Symbol, build_structure};
use crate::wick::{WickProduct, RenormalizedSPDE, Phi4Renormalization};
use crate::rg::{FixedPoint, BetaFunction, CouplingSpace, RGFlow};
use crate::universal::{UniversalClass, classify_universal, critical_exponents, KPZDynamics, AllenCahnDynamics, Phi4Dynamics};
use crate::holders::{HolderRegularity, holder_exponent, needs_renormalization};

/// PLATO agent types with their learning dynamics classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlatoAgentType {
    /// Gradient descent learner (linear update rule).
    GradientDescent,
    /// Natural gradient learner (quadratic correction).
    NaturalGradient,
    /// Mirror descent (nonlinear map to dual space).
    MirrorDescent,
    /// Policy gradient (gradient of expected return).
    PolicyGradient,
    /// Actor-critic (coupled value + policy updates).
    ActorCritic,
    /// Meta-learner (learning rate adaptation).
    MetaLearner,
    /// Ensemble learner (multiple coupled agents).
    Ensemble,
    /// Attention-based learner (nonlocal interaction).
    Attention,
    /// Bayesian updater (posterior belief update).
    Bayesian,
    /// Evolutionary strategist (population-based).
    Evolutionary,
}

impl std::fmt::Display for PlatoAgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PlatoAgentType::GradientDescent => write!(f, "Gradient Descent"),
            PlatoAgentType::NaturalGradient => write!(f, "Natural Gradient"),
            PlatoAgentType::MirrorDescent => write!(f, "Mirror Descent"),
            PlatoAgentType::PolicyGradient => write!(f, "Policy Gradient"),
            PlatoAgentType::ActorCritic => write!(f, "Actor-Critic"),
            PlatoAgentType::MetaLearner => write!(f, "Meta-Learner"),
            PlatoAgentType::Ensemble => write!(f, "Ensemble"),
            PlatoAgentType::Attention => write!(f, "Attention"),
            PlatoAgentType::Bayesian => write!(f, "Bayesian"),
            PlatoAgentType::Evolutionary => write!(f, "Evolutionary"),
        }
    }
}

/// Classification result for a PLATO agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatoClassification {
    /// The agent type.
    pub agent_type: PlatoAgentType,
    /// Whether renormalization is needed.
    pub needs_renormalization: bool,
    /// The singularity class.
    pub singularity: SingularityClass,
    /// The universal class.
    pub universal_class: UniversalClass,
    /// The Hölder regularity of beliefs.
    pub regularity: HolderRegularity,
    /// The associated SPDE.
    pub spde: AgentSPDE,
    /// Recommended Wick ordering order (0 if not needed).
    pub wick_order: usize,
    /// Renormalization description.
    pub description: String,
}

/// Map a PLATO agent type to its SPDE representation.
pub fn agent_to_spde(agent_type: PlatoAgentType, grid_size: usize) -> AgentSPDE {
    match agent_type {
        PlatoAgentType::GradientDescent => AgentSPDE {
            dimension: grid_size,
            spatial_dim: 1,
            diffusion: DMatrix::identity(grid_size, grid_size),
            nonlinearity: NonlinearityType::Linear,
            noise_intensity: 0.1,
            mass: 1.0,
            coupling: 0.0,
        },
        PlatoAgentType::NaturalGradient => AgentSPDE {
            dimension: grid_size,
            spatial_dim: 1,
            diffusion: DMatrix::identity(grid_size, grid_size),
            nonlinearity: NonlinearityType::Quadratic,
            noise_intensity: 0.5,
            mass: 0.5,
            coupling: 0.1,
        },
        PlatoAgentType::MirrorDescent => AgentSPDE {
            dimension: grid_size,
            spatial_dim: 1,
            diffusion: DMatrix::identity(grid_size, grid_size),
            nonlinearity: NonlinearityType::Cubic,
            noise_intensity: 0.3,
            mass: -0.5,
            coupling: 0.5,
        },
        PlatoAgentType::PolicyGradient => AgentSPDE {
            dimension: grid_size,
            spatial_dim: 1,
            diffusion: DMatrix::identity(grid_size, grid_size),
            nonlinearity: NonlinearityType::GradientSquared,
            noise_intensity: 1.0,
            mass: 0.0,
            coupling: 0.5,
        },
        PlatoAgentType::ActorCritic => AgentSPDE {
            dimension: grid_size * 2,
            spatial_dim: 1,
            diffusion: DMatrix::identity(grid_size * 2, grid_size * 2),
            nonlinearity: NonlinearityType::Cubic,
            noise_intensity: 0.8,
            mass: 0.3,
            coupling: 0.7,
        },
        PlatoAgentType::MetaLearner => AgentSPDE {
            dimension: grid_size,
            spatial_dim: 1,
            diffusion: DMatrix::identity(grid_size, grid_size),
            nonlinearity: NonlinearityType::Quadratic,
            noise_intensity: 0.2,
            mass: 0.1,
            coupling: 0.3,
        },
        PlatoAgentType::Ensemble => AgentSPDE {
            dimension: grid_size * 5,
            spatial_dim: 1,
            diffusion: DMatrix::identity(grid_size * 5, grid_size * 5),
            nonlinearity: NonlinearityType::Cubic,
            noise_intensity: 0.5,
            mass: 0.5,
            coupling: 1.0,
        },
        PlatoAgentType::Attention => AgentSPDE {
            dimension: grid_size,
            spatial_dim: 1,
            diffusion: DMatrix::identity(grid_size, grid_size),
            nonlinearity: NonlinearityType::Polynomial(4),
            noise_intensity: 0.4,
            mass: 0.2,
            coupling: 0.2,
        },
        PlatoAgentType::Bayesian => AgentSPDE {
            dimension: grid_size,
            spatial_dim: 1,
            diffusion: DMatrix::identity(grid_size, grid_size),
            nonlinearity: NonlinearityType::Linear,
            noise_intensity: 0.05,
            mass: 2.0,
            coupling: 0.0,
        },
        PlatoAgentType::Evolutionary => AgentSPDE {
            dimension: grid_size * 10,
            spatial_dim: 1,
            diffusion: DMatrix::identity(grid_size * 10, grid_size * 10),
            nonlinearity: NonlinearityType::GradientSquared,
            noise_intensity: 2.0,
            mass: 0.0,
            coupling: 1.0,
        },
    }
}

/// Classify a PLATO agent: determine its singularity, universality class, and
/// whether renormalization is needed.
pub fn classify_agent(agent_type: PlatoAgentType, grid_size: usize) -> PlatoClassification {
    let spde = agent_to_spde(agent_type, grid_size);
    let singularity = classify_singularity(&spde, spde.spatial_dim);
    let universal_class = classify_universal(&spde);
    let regularity = holder_exponent(spde.spatial_dim, spde.nonlinearity);
    let needs_ren = needs_renormalization(&spde);

    let (wick_order, description) = match agent_type {
        PlatoAgentType::GradientDescent => (0, "Linear dynamics: classical solution exists".to_string()),
        PlatoAgentType::NaturalGradient => (2, "Quadratic nonlinearity: needs Wick ordering of u²".to_string()),
        PlatoAgentType::MirrorDescent => (3, "Cubic nonlinearity: Allen-Cahn phase transitions, needs :u³:".to_string()),
        PlatoAgentType::PolicyGradient => (2, "KPZ-type growth dynamics: gradient squared needs renormalization".to_string()),
        PlatoAgentType::ActorCritic => (3, "Coupled cubic dynamics: Φ⁴-type, needs full Wick ordering".to_string()),
        PlatoAgentType::MetaLearner => (2, "Quadratic learning rate adaptation: Wick ordering of u²".to_string()),
        PlatoAgentType::Ensemble => (3, "Multi-agent Φ⁴ critical dynamics: full renormalization needed".to_string()),
        PlatoAgentType::Attention => (4, "Quartic nonlinearity: higher-order Wick ordering required".to_string()),
        PlatoAgentType::Bayesian => (0, "Linear Bayesian updates: no renormalization needed".to_string()),
        PlatoAgentType::Evolutionary => (2, "KPZ growth in population space: gradient squared renormalization".to_string()),
    };

    PlatoClassification {
        agent_type,
        needs_renormalization: needs_ren,
        singularity,
        universal_class,
        regularity,
        spde,
        wick_order,
        description,
    }
}

/// Classify all PLATO agents and return the full report.
pub fn classify_all_agents(grid_size: usize) -> Vec<PlatoClassification> {
    use PlatoAgentType::*;
    let agents = [GradientDescent, NaturalGradient, MirrorDescent, PolicyGradient,
                  ActorCritic, MetaLearner, Ensemble, Attention, Bayesian, Evolutionary];
    agents.iter().map(|&a| classify_agent(a, grid_size)).collect()
}

/// Get only the singular agents (those needing renormalization).
pub fn singular_agents(grid_size: usize) -> Vec<PlatoClassification> {
    classify_all_agents(grid_size)
        .into_iter()
        .filter(|c| c.needs_renormalization)
        .collect()
}

/// Summary statistics of PLATO agent classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatoSummary {
    /// Total agents classified.
    pub total: usize,
    /// Number of singular agents.
    pub singular_count: usize,
    /// Number of regular agents.
    pub regular_count: usize,
    /// Universal class distribution.
    pub class_distribution: std::collections::HashMap<String, usize>,
    /// Maximum Wick order needed.
    pub max_wick_order: usize,
}

impl PlatoSummary {
    /// Compute summary from classifications.
    pub fn from_classifications(classifications: &[PlatoClassification]) -> Self {
        let mut class_dist = std::collections::HashMap::new();
        let mut max_wick = 0;
        let mut singular = 0;

        for c in classifications {
            if c.needs_renormalization {
                singular += 1;
            }
            max_wick = max_wick.max(c.wick_order);
            *class_dist.entry(format!("{}", c.universal_class)).or_insert(0) += 1;
        }

        Self {
            total: classifications.len(),
            singular_count: singular,
            regular_count: classifications.len() - singular,
            class_distribution: class_dist,
            max_wick_order: max_wick,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_to_spde_gradient() {
        let spde = agent_to_spde(PlatoAgentType::GradientDescent, 10);
        assert_eq!(spde.nonlinearity, NonlinearityType::Linear);
    }

    #[test]
    fn test_agent_to_spde_policy_gradient() {
        let spde = agent_to_spde(PlatoAgentType::PolicyGradient, 10);
        assert_eq!(spde.nonlinearity, NonlinearityType::GradientSquared);
    }

    #[test]
    fn test_agent_to_spde_actor_critic() {
        let spde = agent_to_spde(PlatoAgentType::ActorCritic, 10);
        assert_eq!(spde.nonlinearity, NonlinearityType::Cubic);
    }

    #[test]
    fn test_classify_gradient_descent() {
        let c = classify_agent(PlatoAgentType::GradientDescent, 10);
        assert!(!c.needs_renormalization);
        assert_eq!(c.singularity, SingularityClass::Regular);
        assert_eq!(c.wick_order, 0);
    }

    #[test]
    fn test_classify_policy_gradient() {
        let c = classify_agent(PlatoAgentType::PolicyGradient, 10);
        assert!(c.needs_renormalization);
        assert_eq!(c.universal_class, UniversalClass::KPZ);
    }

    #[test]
    fn test_classify_mirror_descent() {
        let c = classify_agent(PlatoAgentType::MirrorDescent, 10);
        assert!(c.needs_renormalization);
        assert_eq!(c.universal_class, UniversalClass::AllenCahn);
    }

    #[test]
    fn test_classify_bayesian() {
        let c = classify_agent(PlatoAgentType::Bayesian, 10);
        assert!(!c.needs_renormalization);
    }

    #[test]
    fn test_classify_ensemble() {
        let c = classify_agent(PlatoAgentType::Ensemble, 10);
        assert!(c.needs_renormalization);
        assert_eq!(c.universal_class, UniversalClass::Phi4);
    }

    #[test]
    fn test_classify_all_agents() {
        let all = classify_all_agents(10);
        assert_eq!(all.len(), 10);
    }

    #[test]
    fn test_singular_agents() {
        let singular = singular_agents(10);
        assert!(singular.len() >= 5); // Most agents are singular
        for c in &singular {
            assert!(c.needs_renormalization);
        }
    }

    #[test]
    fn test_plato_summary() {
        let all = classify_all_agents(10);
        let summary = PlatoSummary::from_classifications(&all);
        assert_eq!(summary.total, 10);
        assert!(summary.singular_count > 0);
        assert!(summary.regular_count > 0);
        assert!(summary.max_wick_order > 0);
    }

    #[test]
    fn test_agent_type_display() {
        assert_eq!(format!("{}", PlatoAgentType::GradientDescent), "Gradient Descent");
        assert_eq!(format!("{}", PlatoAgentType::ActorCritic), "Actor-Critic");
    }

    #[test]
    fn test_classify_attention() {
        let c = classify_agent(PlatoAgentType::Attention, 10);
        assert_eq!(c.wick_order, 4);
    }

    #[test]
    fn test_classify_evolutionary() {
        let c = classify_agent(PlatoAgentType::Evolutionary, 10);
        assert!(c.needs_renormalization);
    }

    #[test]
    fn test_classify_natural_gradient() {
        let c = classify_agent(PlatoAgentType::NaturalGradient, 10);
        assert_eq!(c.wick_order, 2);
    }

    #[test]
    fn test_classify_meta_learner() {
        let c = classify_agent(PlatoAgentType::MetaLearner, 10);
        assert_eq!(c.wick_order, 2);
    }

    #[test]
    fn test_classify_actor_critic() {
        let c = classify_agent(PlatoAgentType::ActorCritic, 10);
        assert_eq!(c.wick_order, 3);
        assert_eq!(c.universal_class, UniversalClass::Phi4);
    }

    #[test]
    fn test_summary_class_distribution() {
        let all = classify_all_agents(10);
        let summary = PlatoSummary::from_classifications(&all);
        assert!(summary.class_distribution.len() >= 2);
    }
}
