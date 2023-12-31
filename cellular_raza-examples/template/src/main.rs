use cellular_raza::building_blocks::prelude::*;
use cellular_raza::concepts::prelude::*;
use cellular_raza::concepts_derive::*;
use cellular_raza::core::backend::chili::*;
use cellular_raza::core_derive::*;

use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
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
    // #[Cycle]
    // pub cycle: NoCycle,
    #[CellularReactions(Nothing, Nothing)]
    pub reactions: NoCellularReactions,
    #[ExtracellularGradient(nalgebra::SVector<Vector2<f64>, 2>)]
    pub gradients: NoExtracellularGradientSensing,
    #[Volume]
    pub volume: Vol,
}

fn main() {}
