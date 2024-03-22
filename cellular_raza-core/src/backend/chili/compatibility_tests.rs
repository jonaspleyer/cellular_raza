//! This module serves to test for compatibility between types and implementations before running a
//! simulation.
//!
//! The rationale behind this approach is to circumvent terse
//! generic error messages which are hard to read.
//! Eventually, we might opt to also implement custom error messages.

#[allow(unused)]
pub fn domain_agents<D, C, S, Ci>(domain: &D, agents: &Ci)
where
    D: cellular_raza_concepts::domain_new::Domain<C, S, Ci>,
    Ci: IntoIterator<Item = C>,
{
}

#[allow(unused)]
pub fn mechanics_interaction<C, Ci, Pos, Vel, For, Inf, Float>(agents: &Ci)
where
    Ci: IntoIterator<Item = C>,
    C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
    C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>,
{
}

#[allow(unused)]
pub fn aux_storage_default<A>(aux_storage: &A)
where
    A: Default,
{
}

#[allow(unused)]
pub fn time_stepper_mechanics<Pos, Vel, For, T, C, Ci, Float>(time_stepper: &T, agents: &Ci)
where
    T: crate::time::TimeStepper<Float>,
    Ci: IntoIterator<Item = C>,
    C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
{
}

#[allow(unused)]
pub fn mechanics_implemented<Pos, Vel, For, Float, C, Ci>(agents: &Ci)
where
    Ci: IntoIterator<Item = C>,
    C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
{
}

#[allow(unused)]
pub fn interaction_implemented<Pos, Vel, For, Inf, C, Ci>(agents: &Ci)
where
    Ci: IntoIterator<Item = C>,
    C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>,
{
}

#[allow(unused)]
pub fn cycle_implemented<C, Ci>(agents: &Ci)
where
    Ci: IntoIterator<Item = C>,
    C: cellular_raza_concepts::Cycle<C>,
{
}

#[allow(unused)]
pub fn reactions_implemented<C, Ci>(agents: &Ci)
where
    Ci: IntoIterator<Item = C>,
    // TODO
{
}

#[allow(unused)]
pub fn reactions_volume<C, Ci>(agents: &Ci)
where
    Ci: IntoIterator<Item = C>,
{
}
