use crate::errors::CalcError;

// TODO Define trait aliases for Position and Force

// TODO use trait alias when available
// pub trait InteractionInformation = Send + Sync + Clone + core::fmt::Debug;
pub trait InteractionInformation: Send + Sync + Clone + core::fmt::Debug {}
impl<T> InteractionInformation for T where T: Send + Sync + Clone + core::fmt::Debug {}

pub trait Interaction<Pos, Vel, Force, Inf = ()> {
    fn get_interaction_information(&self) -> Option<Inf> {
        None
    }

    fn calculate_force_on(
        &self,
        own_pos: &Pos,
        own_vel: &Vel,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_info: &Option<Inf>,
    ) -> Option<Result<Force, CalcError>>;
    // TODO
    // fn contact_function(&mut self, other_cell: &C, environment: &mut Env) -> Result<(), SimulationError>;
}
// TODO we should not use the option here

pub trait InteractionExtracellularGradient<Cell, ConcGradientExtracellular> {
    fn sense_gradient(
        cell: &mut Cell,
        gradient: &ConcGradientExtracellular,
    ) -> Result<(), CalcError>;
}

pub trait CellularReactions<ConcVecIntracellular, ConcVecExtracellular = ConcVecIntracellular> {
    fn get_intracellular(&self) -> ConcVecIntracellular;
    fn set_intracellular(&mut self, concentration_vector: ConcVecIntracellular);
    fn calculate_intra_and_extracellular_reaction_increment(
        &self,
        internal_concentration_vector: &ConcVecIntracellular,
        #[cfg(feature = "fluid_mechanics")] external_concentration_vector: &ConcVecExtracellular,
    ) -> Result<(ConcVecIntracellular, ConcVecExtracellular), CalcError>;
}
