use crate::concepts::cycle::{Cycle,CycleEvent};
use crate::concepts::interaction::{CellularReactions,Interaction,InteractionExtracellularGRadient};
use crate::concepts::mechanics::{Position,Force,Velocity,Mechanics};

use serde::{Serialize,Deserialize};


#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct ModularCell<Pos, Mec, Int, Cyc, React, IntExtracellular>
{
    pub mechanics: MechanicsOptions<Mec, Pos>,
    pub interaction: Int,
    pub interaction_extracellular: IntExtracellular,
    pub cycle: Cyc,
    pub cellular_reactions: React,
}


#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct MechanicsFixedPos<Pos> {
    pub pos: Pos,
}


#[derive(Clone,Debug,Serialize,Deserialize)]
pub enum MechanicsOptions<Mec, Pos> {
    Mechanics(Mec),
    FixedPos(MechanicsFixedPos<Pos>),
}


macro_rules! define_no_cellular_reactions{
    ($conc_vec_intracellular:ty, $conc_vec_extracellular:ty) => {
        #[derive(Clone,Debug,Serialize,Deserialize)]
        pub struct NoCellularreactions {}

        impl<Pos, Mec, Int, Cyc, IntExtracellular> CellularReactions<$conc_vec_intracellular, $conc_vec_extracellular> for ModularCell<Pos, Mec, Int, Cyc, NoCellularreactions, IntExtracellular>
        where
            $conc_vec_intracellular: num::Zero,
            $conc_vec_extracellular: num::Zero,
        {
            fn calculate_intra_and_extracellular_reaction_increment(&self, _internal_concentration_vector: &$conc_vec_intracellular, _external_concentration_vector: &$conc_vec_extracellular) -> Result<($conc_vec_intracellular, $conc_vec_extracellular), crate::concepts::errors::CalcError> {
                Ok((<$conc_vec_intracellular>::zero(), <$conc_vec_extracellular>::zero()))
            }

            fn get_intracellular(&self) -> $conc_vec_intracellular {
                <$conc_vec_intracellular>::zero()
            }

            fn set_intracellular(&mut self, _concentration_vector: $conc_vec_intracellular) {}
        }

    }
}

define_no_cellular_reactions!{f64, f64}


impl<Pos, For, Vel, Mec, Int, Cyc, React, IntExtracellular> Mechanics<Pos, For, Vel> for ModularCell<Pos, Mec, Int, Cyc, React, IntExtracellular>
where
    Mec: Mechanics<Pos, For, Vel>,
    Pos: Position,
    For: Force,
    Vel: Velocity,
{
    fn set_pos(&mut self, pos: &Pos) {
        match &mut self.mechanics {
            MechanicsOptions::Mechanics(mech) => mech.set_pos(pos),
            MechanicsOptions::FixedPos(mfp) => mfp.pos = pos.clone(),
        }
    }

    fn pos(&self) -> Pos {
        match &self.mechanics {
            MechanicsOptions::Mechanics(mech) => mech.pos(),
            MechanicsOptions::FixedPos(mfp) => mfp.pos.clone(),
        }
    }

    fn set_velocity(&mut self, velocity: &Vel) {
        match &mut self.mechanics {
            MechanicsOptions::Mechanics(mech) => mech.set_velocity(velocity),
            MechanicsOptions::FixedPos(_) => (),
        }
    }

    fn velocity(&self) -> Vel {
        match &self.mechanics {
            MechanicsOptions::Mechanics(mech) => mech.velocity(),
            MechanicsOptions::FixedPos(_) => Vel::zero(),
        }
    }

    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), crate::prelude::CalcError> {
        match &self.mechanics {
            MechanicsOptions::Mechanics(mech) => mech.calculate_increment(force),
            MechanicsOptions::FixedPos(mfp) => Ok((mfp.pos.clone() * 0.0, Vel::zero()))
        }
    }
}


impl<Pos, For, Inf, Mec, Int, Cyc, React, IntExtracellular> Interaction<Pos, For, Inf> for ModularCell<Pos, Mec, Int, Cyc, React, IntExtracellular>
where
    Pos: Position,
    For: Force,
    Int: Interaction<Pos, For, Inf>,
{
    fn get_interaction_information(&self) -> Option<Inf> {
        self.interaction.get_interaction_information()
    }

    fn calculate_force_on(&self, own_pos: &Pos, ext_pos: &Pos, ext_information: &Option<Inf>) -> Option<Result<For, crate::prelude::CalcError>> {
        self.interaction.calculate_force_on(own_pos, ext_pos, ext_information)
    }
}


impl<Pos, Mec, Int, Cyc, React, IntExtracellular> Cycle<Self> for ModularCell<Pos, Mec, Int, Cyc, React, IntExtracellular>
where
    Cyc: Cycle<Self>,
{
    fn update_cycle(rng: &mut rand_chacha::ChaCha8Rng, dt: &f64, c: &mut Self) -> Option<CycleEvent> {
        Cyc::update_cycle(rng, dt, c)
    }

    fn divide(rng: &mut rand_chacha::ChaCha8Rng, c: &mut Self) -> Result<Option<Self>, crate::concepts::errors::DivisionError> {
        Cyc::divide(rng, c)
    }
}


impl<Pos, ConcVecIntracellular, ConcVecExtracellular, Mec, Int, Cyc, React, IntExtracellular> CellularReactions<ConcVecIntracellular, ConcVecExtracellular> for ModularCell<Pos, Mec, Int, Cyc, React, IntExtracellular>
where
    React: CellularReactions<ConcVecIntracellular, ConcVecExtracellular>
{
    fn calculate_intra_and_extracellular_reaction_increment(&self, internal_concentration_vector: &ConcVecIntracellular, external_concentration_vector: &ConcVecExtracellular) -> Result<(ConcVecIntracellular, ConcVecExtracellular), crate::prelude::CalcError> {
        self.cellular_reactions.calculate_intra_and_extracellular_reaction_increment(internal_concentration_vector, external_concentration_vector)
    }

    fn get_intracellular(&self) -> ConcVecIntracellular {
        self.cellular_reactions.get_intracellular()
    }

    fn set_intracellular(&mut self, concentration_vector: ConcVecIntracellular) {
        self.cellular_reactions.set_intracellular(concentration_vector)
    }
}


#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct NoExtracellularGradientSensing {}


impl<C, ConcGradientExtracellular> InteractionExtracellularGRadient<C, ConcGradientExtracellular> for NoExtracellularGradientSensing
{
    fn sense_gradient(_cell: &mut C, _extracellular_gradient: &ConcGradientExtracellular) -> Result<(), crate::concepts::errors::CalcError> {
        Ok(())
    }
}


impl<Pos, Mec, Int, Cyc, React, IntExtracellular, ConcGradientExtracellular> InteractionExtracellularGRadient<ModularCell<Pos, Mec, Int, Cyc, React, IntExtracellular>, ConcGradientExtracellular> for ModularCell<Pos, Mec, Int, Cyc, React, IntExtracellular>
where
    IntExtracellular: InteractionExtracellularGRadient<Self, ConcGradientExtracellular>,
{
    fn sense_gradient(cell: &mut Self, extracellular_gradient: &ConcGradientExtracellular) -> Result<(), crate::concepts::errors::CalcError> {
        IntExtracellular::sense_gradient(cell, extracellular_gradient)
    }
}
