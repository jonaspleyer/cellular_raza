use cellular_raza::core::backend::chili::*;
use cellular_raza::prelude::*;

use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
struct MyCycle;

impl Cycle<Agent> for MyCycle {
    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Agent) -> Result<Agent, DivisionError> {
        todo!()
    }

    fn update_cycle(
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Agent,
    ) -> Option<CycleEvent> {
        None
    }
}

#[derive(Deserialize, Serialize)]
struct Vol(f64);

impl Volume for Vol {
    fn get_volume(&self) -> f64 {
        self.0
    }
}

#[derive(CellAgent)] //, Deserialize, Serialize)]
struct Agent {
    #[Mechanics(Vector2<f32>, Vector2<f32>, Vector2<f32>)]
    pub mechanics: NewtonDamped2DF32,
    #[Interaction(Vector2<f64>, Vector2<f64>, Vector2<f64>)]
    pub interaction: BoundLennardJones,
    #[Cycle]
    pub cycle: MyCycle,
    #[CellularReactions(Nothing, Nothing)]
    pub reactions: NoCellularReactions,
    #[ExtracellularGradient(nalgebra::SVector<Vector2<f64>, 2>)]
    pub gradients: NoExtracellularGradientSensing,
    #[Volume]
    pub volume: Vol,
}

fn main() {}
