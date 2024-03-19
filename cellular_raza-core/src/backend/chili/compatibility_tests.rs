#[allow(unused)]
pub fn comp_domain_agents<D, C, S, Ci>(
    domain: &D,
    agents: &Ci,
)
where
    D: cellular_raza_concepts::domain_new::Domain<C, S, Ci>,
    Ci: IntoIterator<Item=C>,
{}

#[allow(unused)]
pub fn comp_mechanics_interaction<C, Ci, Pos, Vel, For, Inf, Float>(
    agents: &Ci
)
where
    Ci: IntoIterator<Item=C>,
    C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
    C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>,
{}
