/// ```
/// use cellular_raza_concepts_derive::CellAgent;
/// use cellular_raza_concepts::*;
/// use rand_chacha::ChaCha8Rng;
/// struct Agent;
/// impl<NA> cellular_raza_concepts::Cycle<NA> for Agent {
///     fn update_cycle(
///         rng: &mut ChaCha8Rng,
///         dt: &f64,
///         cell: &mut NA,
///     ) -> Option<CycleEvent> {
///         None
///     }
///
///     fn divide(
///         rng: &mut ChaCha8Rng,
///         cell: &mut NA,
///     ) -> Result<NA, DivisionError> {
///         unimplemented!()
///     }
/// }
/// #[derive(CellAgent)]
/// struct NewAgent1 {
///     #[Cycle]
///     old_agent: Agent,
/// }
/// #[derive(CellAgent)]
/// struct NewAgent2(#[Cycle]Agent);
///
/// ```
#[doc(hidden)]
#[allow(unused)]
fn derive_cycle() {}

/// ```
/// use cellular_raza_concepts_derive::CellAgent;
/// use cellular_raza_concepts::Position;
/// struct PositionModel;
/// impl Position<u8> for PositionModel {
///     fn pos(&self) -> u8 {
///         1
///     }
///
///     fn set_pos(&mut self, _pos: &u8) {}
/// }
/// #[derive(CellAgent)]
/// struct NewAgent {
///     #[Position]
///     pos_model: PositionModel,
/// }
/// let new_agent = NewAgent {
///     pos_model: PositionModel,
/// };
/// assert_eq!(new_agent.pos(), 1);
/// ```
#[doc(hidden)]
#[allow(unused)]
fn derive_position() {}

/// ```
/// use cellular_raza_concepts_derive::CellAgent;
/// use cellular_raza_concepts::Velocity;
/// struct VelocityModel;
/// impl Velocity<i32> for VelocityModel {
///     fn velocity(&self) -> i32 {
///         1
///     }
///
///     fn set_velocity(&mut self, _velocity: &i32) {}
/// }
/// #[derive(CellAgent)]
/// struct NewAgent {
///     #[Velocity]
///     velocity_model: VelocityModel,
/// }
/// let new_agent = NewAgent {
///     velocity_model: VelocityModel,
/// };
/// assert_eq!(new_agent.velocity(), 1);
/// ```
#[doc(hidden)]
#[allow(unused)]
fn derive_velocity() {}

/// ```
/// use cellular_raza_concepts_derive::CellAgent;
/// use cellular_raza_concepts::*;
/// use rand_chacha::ChaCha8Rng;
/// struct MechanicsModel;
/// impl cellular_raza_concepts::Mechanics<f32, f32, f32, f32> for MechanicsModel {
///     /* fn pos(&self) -> f32 {0.0}
///     fn velocity(&self) -> f32 {0.0}
///     fn set_pos(&mut self, pos: &f32) {}
///     fn set_velocity(&mut self, vel: &f32) {}*/
///     fn calculate_increment(
///         &self,
///         dt: f32
///     ) -> Result<(f32, f32), CalcError> {
///         unimplemented!()
///     }
///
///     fn get_random_contribution(
///         &self,
///         rng: &mut rand_chacha::ChaCha8Rng,
///         dt: f32,
///     ) -> Result<(f32, f32), RngError> {
///         unimplemented!()
///     }
/// }
/// #[derive(CellAgent)]
/// struct NewAgent1 {
///     #[Mechanics]
///     mechanics: MechanicsModel
/// }
// ```
#[doc(hidden)]
#[allow(unused)]
fn derive_mechanics() {}

///```
/// use cellular_raza_concepts_derive::CellAgent;
/// use cellular_raza_concepts::*;
/// use rand_chacha::ChaCha8Rng;
/// struct InteractionModel;
/// impl cellular_raza_concepts::Interaction<f32, f32, f32> for InteractionModel {
///     fn get_interaction_information(&self) -> () {}
///     fn calculate_force_between(
///         &self,
///         own_pos: &f32,
///         ext_pos: &f32,
///         own_vel: &f32,
///         ext_vel: &f32,
///         ext_info: &()
///     ) -> Result<(f32, f32), CalcError> {
///         unimplemented!()
///     }
/// }
/// #[derive(CellAgent)]
/// struct NewAgent1 {
///     #[Interaction]
///     mechanics: InteractionModel,
/// }
// ```
#[doc(hidden)]
#[allow(unused)]
fn derive_interaction() {}
