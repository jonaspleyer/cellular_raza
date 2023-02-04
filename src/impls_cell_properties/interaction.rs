use crate::concepts::errors::CalcError;
use crate::concepts::interaction::*;


use nalgebra::SVector;
use serde::{Serialize,Deserialize};


#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct NoInteraction {}

impl<Pos, For> Interaction<Pos, For> for NoInteraction {
    fn force(&self, _: &Pos, _: &Pos) -> Option<Result<For, CalcError>> {
        return None;
    }
}



#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct LennardJones {
    pub epsilon: f64,
    pub sigma: f64,
}

macro_rules! implement_lennard_jones_nd(
    ($dim:literal) =>  {
        impl Interaction<SVector<f64, $dim>,SVector<f64, $dim>> for LennardJones {
            fn force(&self, own_pos: &SVector<f64, $dim>, ext_pos: &SVector<f64, $dim>) -> Option<Result<SVector<f64, $dim>, CalcError>> {
                let z = own_pos - ext_pos;
                let r = z.norm();
                let dir = z/r;
                let val = 4.0 * self.epsilon / r * (12.0 * (self.sigma/r).powf(12.0) - 1.0 * (self.sigma/r).powf(1.0));
                let max = 10.0 * self.epsilon / r;
                return Some(Ok(dir * max.min(val)));
            }
        }
    }
);


implement_lennard_jones_nd!(1);
implement_lennard_jones_nd!(2);
implement_lennard_jones_nd!(3);
