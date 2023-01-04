use crate::prelude::*;

use nalgebra::SVector;
use serde::{Serialize,Deserialize};

use core::fmt::Debug;


macro_rules! implement_custom_cell {
    ($d:expr, $name:ident) => {
        #[derive(Clone,Debug,Serialize,Deserialize)]
        pub struct $name {
            pub pos: SVector<f64, $d>,
            pub velocity: SVector<f64, $d>,

            pub cell_radius: f64,
            pub potential_strength: f64,

            pub maximum_age: f64,

            pub remove: bool,
            pub current_age: f64,
        }


        impl Cycle<$name> for $name {
            fn update_cycle(dt: &f64, cell: &mut $name) {
                cell.current_age += dt;
                if cell.current_age > cell.maximum_age {
                    cell.remove = true;
                }
            }
        }

        impl Interaction<SVector<f64, $d>, SVector<f64, $d>> for $name {
            fn force(&self, own_pos: &SVector<f64, $d>, ext_pos: &SVector<f64, $d>) -> Option<Result<SVector<f64, $d>, CalcError>> {
                let z = own_pos - ext_pos;
                let r = z.norm();
                let dir = z/r;
                Some(Ok(dir * 0.0_f64.max(self.potential_strength * (self.cell_radius - r))))
            }
        }

        impl Mechanics<SVector<f64, $d>, SVector<f64, $d>, SVector<f64, $d>> for $name {
            fn pos(&self) -> SVector<f64, $d> {
                self.pos
            }

            fn velocity(&self) -> SVector<f64, $d> {
                self.velocity
            }

            fn set_pos(&mut self, p: &SVector<f64, $d>) {
                self.pos = *p;
            }

            fn set_velocity(&mut self, v: &SVector<f64, $d>) {
                self.velocity = *v;
            }

            fn calculate_increment(&self, force: SVector<f64, $d>) -> Result<(SVector<f64, $d>, SVector<f64, $d>), CalcError> {
                let dx = self.velocity;
                let dv = force;
                Ok((dx, dv))
            }
        }
    }
}


implement_custom_cell!(1, CustomCell1D);
implement_custom_cell!(2, CustomCell2D);
implement_custom_cell!(3, CustomCell3D);
