use crate::concepts::cycle::{Cycle,CycleEvent};
use crate::concepts::interaction::Interaction;
use crate::concepts::mechanics::{Position,Force,Velocity,Mechanics};

use serde::{Serialize,Deserialize};


#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct ModularCell<Pos, Mec, Int, Cyc>
{
    pub mechanics: MechanicsOptions<Mec, Pos>,
    pub interaction: Int,
    pub cycle: Cyc,
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


impl<Pos, For, Vel, Mec, Int, Cyc> Mechanics<Pos, For, Vel> for ModularCell<Pos, Mec, Int, Cyc>
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


impl<Pos, For, Inf, Mec, Int, Cyc> Interaction<Pos, For, Inf> for ModularCell<Pos, Mec, Int, Cyc>
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


impl<Pos, Mec, Int, Cyc> Cycle<Self> for ModularCell<Pos, Mec, Int, Cyc>
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
