use core::f32;

use cellular_raza::prelude::*;
use nalgebra::{SVector, SimdPartialOrd, Vector2};
use serde::{Deserialize, Serialize};

pub const N_REACTIONS: usize = 1;
pub type ReactionVector = nalgebra::SVector<f32, N_REACTIONS>;

#[derive(Serialize, Deserialize, Clone, core::fmt::Debug)]
pub struct MyInteraction {
    pub potential_strength: f32,
    pub relative_interaction_range: f32,
    pub cell_radius: f32,
}

impl Interaction<Vector2<f32>, Vector2<f32>, Vector2<f32>, f32> for MyInteraction {
    fn calculate_force_between(
        &self,
        own_pos: &Vector2<f32>,
        _own_vel: &Vector2<f32>,
        ext_pos: &Vector2<f32>,
        _ext_vel: &Vector2<f32>,
        ext_radius: &f32,
    ) -> Result<(Vector2<f32>, Vector2<f32>), CalcError> {
        let min_relative_distance_to_center = 0.3162277660168379;
        let (r, dir) =
            match (own_pos - ext_pos).norm() < self.cell_radius * min_relative_distance_to_center {
                false => {
                    let z = own_pos - ext_pos;
                    let r = z.norm();
                    (r, z.normalize())
                }
                true => {
                    let dir = match own_pos == ext_pos {
                        true => {
                            return Ok((nalgebra::Vector2::zeros(), nalgebra::Vector2::zeros()));
                        }
                        false => (own_pos - ext_pos).normalize(),
                    };
                    let r = self.cell_radius * min_relative_distance_to_center;
                    (r, dir)
                }
            };
        // Introduce Non-dimensional length variable
        let sigma = r / (self.cell_radius + ext_radius);
        let bound = 4.0 + 1.0 / sigma;
        let spatial_cutoff = (1.0
            + (self.relative_interaction_range * (self.cell_radius + ext_radius) - r).signum())
            * 0.5;

        // Calculate the strength of the interaction with correct bounds
        let strength = self.potential_strength
            * ((1.0 / sigma).powf(2.0) - (1.0 / sigma).powf(4.0))
                .min(bound)
                .max(-bound);

        // Calculate only attracting and repelling forces
        let attracting_force = dir * strength.max(0.0) * spatial_cutoff;
        let repelling_force = dir * strength.min(0.0) * spatial_cutoff;

        Ok((
            -repelling_force - attracting_force,
            repelling_force + attracting_force,
        ))
    }

    fn get_interaction_information(&self) -> f32 {
        self.cell_radius
    }
}

impl Cycle<MyAgent, f32> for MyAgent {
    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        _dt: &f32,
        cell: &mut MyAgent,
    ) -> Option<CycleEvent> {
        // If the cell is not at the maximum size let it grow
        if cell.interaction.cell_radius > cell.division_radius {
            return Some(CycleEvent::Division);
        }
        None
    }

    fn divide(
        rng: &mut rand_chacha::ChaCha8Rng,
        c1: &mut MyAgent,
    ) -> Result<MyAgent, DivisionError> {
        // Clone existing cell
        let mut c2 = c1.clone();

        let r = c1.interaction.cell_radius;

        // Make both cells smaller
        // Also keep old cell larger
        c1.interaction.cell_radius /= std::f32::consts::SQRT_2;
        c2.interaction.cell_radius /= std::f32::consts::SQRT_2;

        // Generate cellular splitting direction randomly
        use rand::Rng;
        let angle_1 = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
        let dir_vec = nalgebra::Rotation2::new(angle_1) * nalgebra::Vector2::from([1.0, 0.0]);

        // Define new positions for cells
        // It is randomly chosen if the old cell is left or right
        let offset = dir_vec * r / std::f32::consts::SQRT_2;
        let old_pos = c1.pos();

        c1.set_pos(&(old_pos + offset));
        c2.set_pos(&(old_pos - offset));

        // Decrease the amount of food in the cells
        c1.intracellular_food *= 0.5;
        c2.intracellular_food *= 0.5;

        Ok(c2)
    }
}

// COMPONENT DESCRIPTION
// 0         CELL RADIUS
impl Intracellular<ReactionVector> for MyAgent {
    fn set_intracellular(&mut self, intracellular: ReactionVector) {
        self.interaction.cell_radius = intracellular[0];
    }

    fn get_intracellular(&self) -> ReactionVector {
        [self.interaction.cell_radius].into()
    }
}

impl ReactionsExtra<ReactionVector, ReactionVector> for MyAgent {
    fn calculate_combined_increment(
        &self,
        _intracellular: &ReactionVector,
        extracellular: &ReactionVector,
    ) -> Result<(ReactionVector, ReactionVector), CalcError> {
        let extra = extracellular;
        let u = self.uptake_rate;

        let uptake = u * extra;

        let incr_intra: ReactionVector = [self.growth_rate * uptake[0]].into();
        let incr_extra = -uptake;

        Ok((incr_intra, incr_extra))
    }
}

#[derive(Clone, Serialize, Deserialize, CellAgent)]
pub struct MyAgent {
    #[Mechanics]
    pub mechanics: NewtonDamped2DF32,
    #[Interaction]
    pub interaction: MyInteraction,
    pub intracellular_food: f32,
    pub uptake_rate: f32,
    pub division_radius: f32,
    pub growth_rate: f32,
}

#[derive(Domain, Clone)]
pub struct MyDomain {
    #[DomainRngSeed]
    #[DomainPartialDerive]
    #[SortCells]
    pub domain: CartesianCuboid<f32, 2>,
    pub reactions_dx: f32,
    pub diffusion_constant: f32,
    pub initial_value: ReactionVector,
}

#[derive(SubDomain, Clone, Serialize)]
pub struct MySubDomain {
    #[Base]
    #[SortCells]
    #[Mechanics]
    pub subdomain: CartesianSubDomain<f32, 2>,
    pub reactions_min: nalgebra::Vector2<f32>,
    pub reactions_max: nalgebra::Vector2<f32>,
    pub reactions_dx: nalgebra::Vector2<f32>,
    pub extracellular: ndarray::Array3<f32>,
    pub diffusion_constant: f32,
    increments: [ndarray::Array3<f32>; 3],
    increments_start: usize,
    helper: ndarray::Array3<f32>,
}

impl MySubDomain {
    fn merge_values(
        &mut self,
        neighbor: Result<NeighborValue, CalcError>,
    ) -> Result<(), CalcError> {
        let neighbor = neighbor?;
        let neighbor_slice = neighbor.extracellular_slice;
        self.helper
            .slice_mut(ndarray::s![
                neighbor.index_min[0]..neighbor.index_max[0],
                neighbor.index_min[1]..neighbor.index_max[1],
                ..
            ])
            .assign(&neighbor_slice);
        Ok(())
    }

    pub fn get_extracellular_index_raw(
        reactions_min: &nalgebra::Vector2<f32>,
        reactions_max: &nalgebra::Vector2<f32>,
        reactions_dx: &nalgebra::Vector2<f32>,
        pos: &nalgebra::Vector2<f32>,
    ) -> Result<nalgebra::Vector2<usize>, CalcError> {
        if pos[0] < reactions_min[0]
            || pos[0] > reactions_max[0]
            || pos[1] < reactions_min[1]
            || pos[1] > reactions_max[1]
        {
            return Err(CalcError(format!(
                "position {:?} is not contained in domain with boundaries {:?} {:?}",
                pos, reactions_min, reactions_max
            )));
        }
        let index = (pos - reactions_min)
            .component_div(&reactions_dx)
            .map(|i| i as usize);
        Ok(index)
    }

    pub fn get_extracellular_index(
        &self,
        pos: &nalgebra::Vector2<f32>,
    ) -> Result<nalgebra::Vector2<usize>, CalcError> {
        Self::get_extracellular_index_raw(
            &self.reactions_min,
            &self.reactions_max,
            &self.reactions_dx,
            &pos,
        )
    }
}

pub struct NeighborValue {
    index_min: nalgebra::Vector2<usize>,
    index_max: nalgebra::Vector2<usize>,
    extracellular_slice: ndarray::Array3<f32>,
}

pub struct BorderInfo {
    min_sent: nalgebra::Vector2<f32>,
    max_sent: nalgebra::Vector2<f32>,
}

impl SubDomainReactions<nalgebra::SVector<f32, 2>, ReactionVector, f32> for MySubDomain {
    type NeighborValue = Result<NeighborValue, CalcError>;
    type BorderInfo = BorderInfo;

    fn treat_increments<I, J>(&mut self, neighbors: I, sources: J) -> Result<(), CalcError>
    where
        I: IntoIterator<Item = Self::NeighborValue>,
        J: IntoIterator<Item = (nalgebra::SVector<f32, 2>, ReactionVector)>,
    {
        use core::ops::AddAssign;
        use ndarray::s;
        let dx = self.reactions_dx[0];
        let dy = self.reactions_dx[1];
        let dd2 = dx.powf(-2.0) + dy.powf(-2.0);

        // Helper variable to store current concentrations
        let co = &self.extracellular;

        // Use helper array which is +2 in every spatial dimension larger than the original array
        // We do this to seamlessly incorporate boundary conditions
        self.helper.fill(0.0);
        // Fill inner part of the array
        // _ _ _ _ _ _ _
        // _ x x x x x _
        // _ x x x x x _
        // _ _ _ _ _ _ _
        self.helper.slice_mut(s![1..-1, 1..-1, ..]).assign(&co);

        // First assume that we obey neumann-boundary conditions and fill outer parts accordingly
        // _ x x x x x _
        // x _ _ _ _ _ x
        // x _ _ _ _ _ x
        // _ x x x x x _
        let co = &self.extracellular;
        self.helper
            .slice_mut(s![0, 1..-1, ..])
            .assign(&co.slice(s![0, .., ..]));
        self.helper
            .slice_mut(s![-1, 1..-1, ..])
            .assign(&co.slice(s![-1, .., ..]));
        self.helper
            .slice_mut(s![1..-1, 0, ..])
            .assign(&co.slice(s![.., 0, ..]));
        self.helper
            .slice_mut(s![1..-1, -1, ..])
            .assign(&co.slice(s![.., -1, ..]));

        // Now overwrite previous assumptions with values from neighbors
        for neighbor in neighbors.into_iter() {
            self.merge_values(neighbor)?;
        }

        // Set increment to next time-step to 0.0 everywhere
        let n_incr = self.increments.len();
        self.increments_start = (self.increments_start + n_incr - 1) % n_incr;
        let start = self.increments_start;
        self.increments[start].fill(0.0);

        // Calculate diffusion part
        self.increments[start].assign(&(-2.0 * dd2 * &self.helper.slice(s![1..-1, 1..-1, ..])));
        self.increments[start]
            .add_assign(&(dx.powf(-2.0) * &self.helper.slice(s![2.., 1..-1, ..])));
        self.increments[start]
            .add_assign(&(dx.powf(-2.0) * &self.helper.slice(s![0..-2, 1..-1, ..])));
        self.increments[start]
            .add_assign(&(dy.powf(-2.0) * &self.helper.slice(s![1..-1, 2.., ..])));
        self.increments[start]
            .add_assign(&(dy.powf(-2.0) * &self.helper.slice(s![1..-1, 0..-2, ..])));

        for (pos, dextra) in sources {
            let index = self.get_extracellular_index(&pos)?;
            let dextra: [f32; N_REACTIONS] = dextra.into();
            self.increments[start]
                .slice_mut(ndarray::s![index[0], index[1], ..])
                .scaled_add(1.0, &ndarray::Array1::<f32>::from_iter(dextra));
        }

        Ok(())
    }

    fn update_fluid_dynamics(&mut self, dt: f32) -> Result<(), cellular_raza::concepts::CalcError> {
        use core::ops::AddAssign;
        let k1 = 5.0 / 12.0;
        let k2 = 8.0 / 12.0;
        let k3 = -1.0 / 12.0;
        let start = self.increments_start;
        let n_incr = self.increments.len();
        self.extracellular.add_assign(
            &(k1 * self.diffusion_constant * dt * &self.increments[start]
                + k2 * self.diffusion_constant * dt * &self.increments[(start + 1) % n_incr]
                + k3 * self.diffusion_constant * dt * &self.increments[(start + 2) % n_incr]),
        );
        self.extracellular.map_inplace(|x| *x = x.max(0.0));
        Ok(())
    }

    fn get_extracellular_at_pos(&self, pos: &SVector<f32, 2>) -> Result<ReactionVector, CalcError> {
        let index = self.get_extracellular_index(pos)?;
        Ok(ReactionVector::from_fn(|i, _| {
            self.extracellular[(index[0], index[1], i)]
        }))
    }

    fn get_neighbor_value(&self, border_info: Self::BorderInfo) -> Self::NeighborValue {
        // Calculate the intersection of both boxes
        let (intersection_min, intersection_max) = (
            self.reactions_min
                .zip_map(&border_info.min_sent, |x, y| x.max(y)),
            self.reactions_max
                .zip_map(&border_info.max_sent, |x, y| x.min(y)),
        );
        let intersection_min_padded =
            (intersection_min - self.reactions_dx).zip_map(&self.reactions_min, |x, y| x.max(y));
        let intersection_max_padded =
            (intersection_max + self.reactions_dx).zip_map(&self.reactions_max, |x, y| x.min(y));
        let ind_min_self = Self::get_extracellular_index_raw(
            &self.reactions_min,
            &self.reactions_max,
            &self.reactions_dx,
            &intersection_min_padded,
        )?;
        let ind_max_self = Self::get_extracellular_index_raw(
            &self.reactions_min,
            &self.reactions_max,
            &self.reactions_dx,
            &intersection_max_padded,
        )?;
        let ind_min_other = Self::get_extracellular_index_raw(
            &(border_info.min_sent - self.reactions_dx),
            &(border_info.max_sent + self.reactions_dx),
            &self.reactions_dx,
            &intersection_min_padded,
        )?;
        let ind_max_other = Self::get_extracellular_index_raw(
            &(border_info.min_sent - self.reactions_dx),
            &(border_info.max_sent + self.reactions_dx),
            &self.reactions_dx,
            &intersection_max_padded,
        )?;
        let extracellular_slice = self
            .extracellular
            .slice(ndarray::s![
                ind_min_self[0]..ind_max_self[0],
                ind_min_self[1]..ind_max_self[1],
                ..
            ])
            .to_owned();

        Ok(NeighborValue {
            index_min: ind_min_other,
            index_max: ind_max_other,
            extracellular_slice,
        })
    }

    fn get_border_info(&self) -> Self::BorderInfo {
        BorderInfo {
            min_sent: self.reactions_min,
            max_sent: self.reactions_max,
        }
    }
}

impl DomainCreateSubDomains<MySubDomain> for MyDomain {
    type SubDomainIndex = usize;
    type VoxelIndex = [usize; 2];

    fn create_subdomains(
        &self,
        n_subdomains: core::num::NonZeroUsize,
    ) -> Result<
        impl IntoIterator<Item = (Self::SubDomainIndex, MySubDomain, Vec<Self::VoxelIndex>)>,
        DecomposeError,
    > {
        let dx = self.reactions_dx;
        let diffusion_constant = self.diffusion_constant;

        // Calculate lattice points for subdomains
        // Introduce padding on the outside of the simulated domain.
        let min = self.domain.get_min();
        let max = self.domain.get_max();
        let nrows = ((max[0] - min[0]) / dx).ceil() as usize;
        let ncols = ((max[1] - min[1]) / dx).ceil() as usize;

        let extracellular_total =
            ndarray::Array3::from_shape_fn((nrows, ncols, N_REACTIONS), |(_, _, n)| {
                self.initial_value[n]
            });

        Ok(self
            .domain
            .create_subdomains(n_subdomains)?
            .into_iter()
            .map(move |(index, subdomain, voxels)| {
                let n_min = (subdomain.get_min() - subdomain.get_domain_min()) / dx;
                let n_max = (subdomain.get_max() - subdomain.get_domain_min()) / dx;
                let n_min = [
                    (n_min[0].floor() as usize).max(0),
                    (n_min[1].floor() as usize).max(0),
                ];
                let n_max = [
                    (n_max[0].ceil() as usize).min(nrows),
                    (n_max[1].ceil() as usize).min(ncols),
                ];

                let extracellular = extracellular_total
                    .slice(ndarray::s![n_min[0]..n_max[0], n_min[1]..n_max[1], ..])
                    .into_owned();
                let reactions_min = [n_min[0] as f32 * dx, n_min[1] as f32 * dx].into();
                let reactions_max = [n_max[0] as f32 * dx, n_max[1] as f32 * dx].into();
                let reactions_dx = [dx; 2].into();

                let sh = extracellular.shape();
                let increment = ndarray::Array3::zeros((sh[0], sh[1], sh[2]));
                let helper = ndarray::Array3::zeros((sh[0] + 2, sh[1] + 2, sh[2]));
                (
                    index,
                    MySubDomain {
                        subdomain,
                        reactions_min,
                        reactions_max,
                        reactions_dx,
                        extracellular,
                        diffusion_constant,
                        increments_start: 0,
                        increments: [increment.clone(), increment.clone(), increment.clone()],
                        helper,
                    },
                    voxels,
                )
            }))
    }
}
