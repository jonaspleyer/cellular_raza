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
/// use cellular_raza_concepts::*;
/// use rand_chacha::ChaCha8Rng;
/// struct MechanicsModel;
/// impl cellular_raza_concepts::Mechanics<f32, f32, f32, f32> for MechanicsModel {
///     fn pos(&self) -> f32 {0.0}
///     fn velocity(&self) -> f32 {0.0}
///     fn set_pos(&mut self, pos: &f32) {}
///     fn set_velocity(&mut self, vel: &f32) {}
///     fn calculate_increment(
///         &self,
///         dt: f32
///     ) -> Result<(f32, f32), CalcError> {
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

// TODO test derivation of more concepts!
