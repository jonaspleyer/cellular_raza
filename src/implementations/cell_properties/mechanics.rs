use crate::concepts::errors::CalcError;
use crate::concepts::mechanics::Mechanics;

use nalgebra::SVector;

use serde::{Serialize,Deserialize};


macro_rules! implement_mechanics_model_nd(
    ($model_name:ident, $dim:literal) => {
        #[derive(Clone,Debug,Serialize,Deserialize)]
        pub struct $model_name {
            pub pos: SVector<f64, $dim>,
            pub vel: SVector<f64, $dim>,
            pub dampening_constant: f64,
        }

        impl Mechanics<SVector<f64, $dim>, SVector<f64, $dim>, SVector<f64, $dim>> for $model_name {
            fn pos(&self) -> SVector<f64, $dim> {
                self.pos
            }

            fn velocity(&self) -> SVector<f64, $dim> {
                self.vel
            }

            fn set_pos(&mut self, p: &SVector<f64, $dim>) {
                self.pos = *p;
            }

            fn set_velocity(&mut self, v: &SVector<f64, $dim>) {
                self.vel = *v;
            }

            fn calculate_increment(&self, force: SVector<f64, $dim>) -> Result<(SVector<f64, $dim>, SVector<f64, $dim>), CalcError> {
                let dx = self.vel;
                let dv = force - self.dampening_constant * self.vel;
                Ok((dx, dv))
            }
        }
    }
);


implement_mechanics_model_nd!(MechanicsModel1D, 1);
implement_mechanics_model_nd!(MechanicsModel2D, 2);
implement_mechanics_model_nd!(MechanicsModel3D, 3);
