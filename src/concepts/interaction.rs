use crate::concepts::errors::CalcError;

// TODO Define trait aliases for Position and Force

pub trait InteractionInformation = Send + Sync + Clone + core::fmt::Debug;


pub trait Interaction<Pos, Force, Inf=()> {
    fn get_interaction_information(&self) -> Option<Inf> {
        None
    }

    fn calculate_force_on(&self, own_pos: &Pos, ext_pos: &Pos, ext_info: &Option<Inf>) -> Option<Result<Force, CalcError>>;
    // TODO
    // fn contact_function(&mut self, other_cell: &C, environment: &mut Env) -> Result<(), SimulationError>;
}
