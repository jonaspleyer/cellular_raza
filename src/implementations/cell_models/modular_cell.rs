use crate::concepts::cycle::{Cycle, CycleEvent};
use crate::concepts::errors::CalcError;
use crate::concepts::interaction::{
    CellularReactions, Interaction, InteractionExtracellularGradient,
};
use crate::concepts::mechanics::{Force, Mechanics, Position, Velocity};

use serde::{Deserialize, Serialize};

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
                crate::concepts::errors::CalcError,
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

define_no_cellular_reactions! {f64, f64}

impl<Pos, For, Vel, Mec, Int, Cyc, React, IntExtracellular> Mechanics<Pos, For, Vel>
    for ModularCell<Mec, Int, Cyc, React, IntExtracellular>
where
    Mec: Mechanics<Pos, For, Vel>,
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

    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError> {
        self.mechanics.calculate_increment(force)
    }
}

impl<Pos, For, Inf, Mec, Int, Cyc, React, IntExtracellular> Interaction<Pos, For, Inf>
    for ModularCell<Mec, Int, Cyc, React, IntExtracellular>
where
    Pos: Position,
    For: Force,
    Int: Interaction<Pos, For, Inf>,
{
    fn get_interaction_information(&self) -> Option<Inf> {
        self.interaction.get_interaction_information()
    }

    fn calculate_force_on(
        &self,
        own_pos: &Pos,
        ext_pos: &Pos,
        ext_information: &Option<Inf>,
    ) -> Option<Result<For, CalcError>> {
        self.interaction
            .calculate_force_on(own_pos, ext_pos, ext_information)
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
        c: &mut Self,
    ) -> Option<CycleEvent> {
        Cyc::update_cycle(rng, dt, c)
    }

    fn divide(
        rng: &mut rand_chacha::ChaCha8Rng,
        c: &mut Self,
    ) -> Result<Option<Self>, crate::concepts::errors::DivisionError> {
        Cyc::divide(rng, c)
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
    ) -> Result<(), crate::concepts::errors::CalcError> {
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
    ) -> Result<(), crate::concepts::errors::CalcError> {
        IntExtracellular::sense_gradient(cell, extracellular_gradient)
    }
}
