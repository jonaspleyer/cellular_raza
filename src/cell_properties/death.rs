use serde::{Serialize,Deserialize};


#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct DeathModel {
    pub release_fluid: bool,
    pub fluid_fraction: f64,
}
