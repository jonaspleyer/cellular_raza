use cellular_raza_concepts::reactions_old::*;
use cellular_raza_concepts::*;

use serde::{Deserialize, Serialize};

use num::Zero;

/// Superseded by the [CellAgent] derive macro.
///
/// The [ModularCell] is a struct with fields that implement the various
/// [concepts](cellular_raza_concepts). The concepts are afterwards derived automatically for the
/// [ModularCell] struct.

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModularCell<Mec, Int, Cyc, React, IntExtracellular> {
    /// Physical mechanics of the cell
    pub mechanics: Mec,
    /// Physical interactions with other cells
    pub interaction: Int,
    /// Interaction with extracellular gradient
    pub interaction_extracellular: IntExtracellular,
    /// Cell cycle
    pub cycle: Cyc,
    /// Intracellular reactions
    pub cellular_reactions: React,
    /// Volume of the cell
    pub volume: f64,
}

/// Do not model intracellular reactions at all.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoCellularReactions;

impl CellularReactions<Nothing, Nothing> for NoCellularReactions
where
    Nothing: num::Zero,
    Nothing: num::Zero,
{
    fn calculate_intra_and_extracellular_reaction_increment(
        &self,
        _internal_concentration_vector: &Nothing,
        _external_concentration_vector: &Nothing,
    ) -> Result<(Nothing, Nothing), CalcError> {
        Ok((<Nothing>::zero(), <Nothing>::zero()))
    }

    fn get_intracellular(&self) -> Nothing {
        <Nothing>::zero()
    }

    fn set_intracellular(&mut self, _concentration_vector: Nothing) {}
}

/// Type alias used when not wanting to simulate any cellular reactions for example.
pub type Nothing = nalgebra::SVector<f64, 0>;

impl<Pos, Vel, For, Float, Mec, Int, Cyc, React, IntExtracellular> Mechanics<Pos, Vel, For, Float>
    for ModularCell<Mec, Int, Cyc, React, IntExtracellular>
where
    Mec: Mechanics<Pos, Vel, For, Float>,
{
    fn get_random_contribution(
        &self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: Float,
    ) -> Result<(Pos, Vel), RngError> {
        self.mechanics.get_random_contribution(rng, dt)
    }

    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError> {
        self.mechanics.calculate_increment(force)
    }
}

impl<Pos, Mec, Int, Cyc, React, InteractionExtracellular> cellular_raza_concepts::Position<Pos>
    for ModularCell<Mec, Int, Cyc, React, InteractionExtracellular>
where
    Mec: Position<Pos>,
{
    fn set_pos(&mut self, pos: &Pos) {
        self.mechanics.set_pos(pos)
    }

    fn pos(&self) -> Pos {
        self.mechanics.pos()
    }
}

impl<Vel, Mec, Int, Cyc, React, InteractionExtracellular> Velocity<Vel>
    for ModularCell<Mec, Int, Cyc, React, InteractionExtracellular>
where
    Mec: Velocity<Vel>,
{
    fn set_velocity(&mut self, velocity: &Vel) {
        self.mechanics.set_velocity(velocity);
    }

    fn velocity(&self) -> Vel {
        self.mechanics.velocity()
    }
}

impl<Inf, Mec, Int, Cyc, React, IntExtracellular> InteractionInformation<Inf>
    for ModularCell<Mec, Int, Cyc, React, IntExtracellular>
where
    Int: InteractionInformation<Inf>,
{
    fn get_interaction_information(&self) -> Inf {
        self.interaction.get_interaction_information()
    }
}

impl<Pos, Vel, For, Inf, Mec, Int, Cyc, React, IntExtracellular> Interaction<Pos, Vel, For, Inf>
    for ModularCell<Mec, Int, Cyc, React, IntExtracellular>
where
    Int: Interaction<Pos, Vel, For, Inf>,
{
    fn calculate_force_between(
        &self,
        own_pos: &Pos,
        own_vel: &Vel,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_information: &Inf,
    ) -> Result<(For, For), CalcError> {
        self.interaction.calculate_force_between(
            own_pos,
            own_vel,
            ext_pos,
            ext_vel,
            ext_information,
        )
    }
}

impl<Mec, Int, Cyc, React, IntExtracellular> Volume
    for ModularCell<Mec, Int, Cyc, React, IntExtracellular>
{
    fn get_volume(&self) -> f64 {
        self.volume
    }
}

impl<Mec, Int, Cyc, Float, React, IntExtracellular> Cycle<Self, Float>
    for ModularCell<Mec, Int, Cyc, React, IntExtracellular>
where
    Cyc: Cycle<Self, Float>,
{
    fn update_cycle(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &Float,
        cell: &mut Self,
    ) -> Option<CycleEvent> {
        Cyc::update_cycle(rng, dt, cell)
    }

    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Self) -> Result<Self, DivisionError> {
        Cyc::divide(rng, cell)
    }

    fn update_conditional_phased_death(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &Float,
        cell: &mut Self,
    ) -> Result<bool, DeathError> {
        Cyc::update_conditional_phased_death(rng, dt, cell)
    }
}

impl<ConcVecIntracellular, ConcVecExtracellular, Mec, Int, Cyc, React, IntExtracellular>
    CellularReactions<ConcVecIntracellular, ConcVecExtracellular>
    for ModularCell<Mec, Int, Cyc, React, IntExtracellular>
where
    React: CellularReactions<ConcVecIntracellular, ConcVecExtracellular>,
{
    fn calculate_intra_and_extracellular_reaction_increment(
        &self,
        internal_concentration_vector: &ConcVecIntracellular,
        external_concentration_vector: &ConcVecExtracellular,
    ) -> Result<(ConcVecIntracellular, ConcVecExtracellular), CalcError> {
        self.cellular_reactions
            .calculate_intra_and_extracellular_reaction_increment(
                internal_concentration_vector,
                external_concentration_vector,
            )
    }

    fn get_intracellular(&self) -> ConcVecIntracellular {
        self.cellular_reactions.get_intracellular()
    }

    fn set_intracellular(&mut self, concentration_vector: ConcVecIntracellular) {
        self.cellular_reactions
            .set_intracellular(concentration_vector)
    }
}

/// Type which allows to simply not model gradients.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoExtracellularGradientSensing;

impl<C, ConcGradientExtracellular> InteractionExtracellularGradient<C, ConcGradientExtracellular>
    for NoExtracellularGradientSensing
{
    fn sense_gradient(
        _cell: &mut C,
        _extracellular_gradient: &ConcGradientExtracellular,
    ) -> Result<(), CalcError> {
        Ok(())
    }
}

impl<Mec, Int, Cyc, React, IntExtracellular, ConcGradientExtracellular>
    InteractionExtracellularGradient<
        ModularCell<Mec, Int, Cyc, React, IntExtracellular>,
        ConcGradientExtracellular,
    > for ModularCell<Mec, Int, Cyc, React, IntExtracellular>
where
    IntExtracellular: InteractionExtracellularGradient<Self, ConcGradientExtracellular>,
{
    fn sense_gradient(
        cell: &mut Self,
        extracellular_gradient: &ConcGradientExtracellular,
    ) -> Result<(), CalcError> {
        IntExtracellular::sense_gradient(cell, extracellular_gradient)
    }
}
