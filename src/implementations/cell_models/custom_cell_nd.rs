use crate::concepts::{
    cycle::{Cycle, CycleEvent},
    errors::{CalcError, DivisionError},
    interaction::Interaction,
    mechanics::Mechanics,
};

use nalgebra::SVector;
use serde::{Deserialize, Serialize};

use core::fmt::Debug;

macro_rules! implement_custom_cell {
    ($d:expr, $name:ident) => {
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
        pub struct $name {
            pub pos: SVector<f64, $d>,
            pub velocity: SVector<f64, $d>,

            pub cell_radius: f64,
            pub potential_strength: f64,

            pub velocity_reduction: f64,

            // Cell cycle, division and death
            pub maximum_age: f64,

            pub remove: bool,
            pub current_age: f64,
        }

        impl Cycle<$name> for $name {
            fn update_cycle(
                _rng: &mut rand_chacha::ChaCha8Rng,
                dt: &f64,
                cell: &mut $name,
            ) -> Option<CycleEvent> {
                cell.current_age += dt;
                if cell.current_age > cell.maximum_age {
                    cell.remove = true;
                }
                None
            }

            fn divide(
                _rng: &mut rand_chacha::ChaCha8Rng,
                _c: &mut $name,
            ) -> Result<Option<$name>, DivisionError> {
                panic!("This function should never be called!");
            }
        }

        impl Interaction<SVector<f64, $d>, SVector<f64, $d>, ()> for $name {
            fn get_interaction_information(&self) -> Option<()> {
                None
            }

            fn calculate_force_on(
                &self,
                own_pos: &SVector<f64, $d>,
                ext_pos: &SVector<f64, $d>,
                _ext_information: &Option<()>,
            ) -> Option<Result<SVector<f64, $d>, CalcError>> {
                let z = own_pos - ext_pos;
                let r = z.norm();
                let sigma = 2.0 * self.cell_radius;
                let spatial_cutoff = (1.0 + (2.0 * sigma - r).signum()) * 0.5;
                let dir = z / r;
                let bound = 4.0 + sigma / r;
                Some(Ok(dir
                    * self.potential_strength
                    * ((sigma / r).powf(2.0) - (sigma / r).powf(4.0))
                        .min(bound)
                        .max(-bound)
                    * spatial_cutoff))
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

            fn calculate_increment(
                &self,
                force: SVector<f64, $d>,
            ) -> Result<(SVector<f64, $d>, SVector<f64, $d>), CalcError> {
                let dx = self.velocity;
                let dv = force - self.velocity_reduction * self.velocity;
                Ok((dx, dv))
            }
        }
    };
}

implement_custom_cell!(1, CustomCell1D);
implement_custom_cell!(2, CustomCell2D);
implement_custom_cell!(3, CustomCell3D);
