use cellular_raza_concepts::cycle::{Cycle, CycleEvent};
use cellular_raza_concepts::errors::{CalcError, RngError};
use cellular_raza_concepts::interaction::{
    CellularReactions, Interaction, InteractionExtracellularGradient,
};
use cellular_raza_concepts::mechanics::{Force, Mechanics, Position, Velocity};

use serde::{Deserialize, Serialize};

use num::Zero;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModularCell<Mec, Int, Cyc, React, IntExtracellular> {
    pub mechanics: Mec,
    pub interaction: Int,
    pub interaction_extracellular: IntExtracellular,
    pub cycle: Cyc,
    pub cellular_reactions: React,
}

macro_rules! define_no_cellular_reactions {
    ($conc_vec_intracellular:ty, $conc_vec_extracellular:ty) => {
        #[derive(Clone, Debug, Serialize, Deserialize)]
        pub struct NoCellularreactions {}

        impl<Mec, Int, Cyc, IntExtracellular>
            CellularReactions<$conc_vec_intracellular, $conc_vec_extracellular>
            for ModularCell<Mec, Int, Cyc, NoCellularreactions, IntExtracellular>
        where
            $conc_vec_intracellular: num::Zero,
            $conc_vec_extracellular: num::Zero,
        {
            fn calculate_intra_and_extracellular_reaction_increment(
                &self,
                _internal_concentration_vector: &$conc_vec_intracellular,
                _external_concentration_vector: &$conc_vec_extracellular,
            ) -> Result<
                ($conc_vec_intracellular, $conc_vec_extracellular),
                cellular_raza_concepts::errors::CalcError,
            > {
                Ok((
                    <$conc_vec_intracellular>::zero(),
                    <$conc_vec_extracellular>::zero(),
                ))
            }

            fn get_intracellular(&self) -> $conc_vec_intracellular {
                <$conc_vec_intracellular>::zero()
            }

            fn set_intracellular(&mut self, _concentration_vector: $conc_vec_intracellular) {}
        }
    };
}

define_no_cellular_reactions! {nalgebra::SVector<f64, 1>, nalgebra::SVector<f64, 1>}

impl<Pos, Vel, For, Mec, Int, Cyc, React, IntExtracellular> Mechanics<Pos, Vel, For>
    for ModularCell<Mec, Int, Cyc, React, IntExtracellular>
where
    Mec: Mechanics<Pos, Vel, For>,
    Pos: Position,
    For: Force,
    Vel: Velocity,
{
    fn set_pos(&mut self, pos: &Pos) {
        self.mechanics.set_pos(pos)
    }

    fn pos(&self) -> Pos {
        self.mechanics.pos()
    }

    fn set_velocity(&mut self, velocity: &Vel) {
        self.mechanics.set_velocity(velocity);
    }

    fn velocity(&self) -> Vel {
        self.mechanics.velocity()
    }

    fn set_random_variable(
        &mut self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: f64,
    ) -> Result<Option<f64>, RngError> {
        self.mechanics.set_random_variable(rng, dt)
    }

    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError> {
        self.mechanics.calculate_increment(force)
    }
}

impl<Pos, Vel, For, Inf, Mec, Int, Cyc, React, IntExtracellular> Interaction<Pos, Vel, For, Inf>
    for ModularCell<Mec, Int, Cyc, React, IntExtracellular>
where
    Pos: Position,
    For: Force,
    Int: Interaction<Pos, Vel, For, Inf>,
{
    fn get_interaction_information(&self) -> Inf {
        self.interaction.get_interaction_information()
    }

    fn calculate_force_between(
        &self,
        own_pos: &Pos,
        own_vel: &Vel,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_information: &Inf,
    ) -> Option<Result<For, CalcError>> {
        self.interaction.calculate_force_between(
            own_pos,
            own_vel,
            ext_pos,
            ext_vel,
            ext_information,
        )
    }
}

impl<Mec, Int, Cyc, React, IntExtracellular> Cycle<Self>
    for ModularCell<Mec, Int, Cyc, React, IntExtracellular>
where
    Cyc: Cycle<Self>,
{
    fn update_cycle(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Self,
    ) -> Option<CycleEvent> {
        Cyc::update_cycle(rng, dt, cell)
    }

    fn divide(
        rng: &mut rand_chacha::ChaCha8Rng,
        cell: &mut Self,
    ) -> Result<Self, cellular_raza_concepts::errors::DivisionError> {
        Cyc::divide(rng, cell)
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoExtracellularGradientSensing {}

impl<C, ConcGradientExtracellular> InteractionExtracellularGradient<C, ConcGradientExtracellular>
    for NoExtracellularGradientSensing
{
    fn sense_gradient(
        _cell: &mut C,
        _extracellular_gradient: &ConcGradientExtracellular,
    ) -> Result<(), cellular_raza_concepts::errors::CalcError> {
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
    ) -> Result<(), cellular_raza_concepts::errors::CalcError> {
        IntExtracellular::sense_gradient(cell, extracellular_gradient)
    }
}
