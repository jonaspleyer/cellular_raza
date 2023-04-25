use crate::concepts::errors::*;

use std::marker::{Send, Sync};

use core::cmp::Eq;
use core::hash::Hash;
use std::ops::{Add, Mul};

use num::Zero;
use serde::{Deserialize, Serialize};

pub trait Domain<C, I, V>: Send + Sync + Serialize + for<'a> Deserialize<'a> {
    fn apply_boundary(&self, cell: &mut C) -> Result<(), BoundaryError>;
    fn get_neighbor_voxel_indices(&self, index: &I) -> Vec<I>;
    fn get_voxel_index(&self, cell: &C) -> I;
    fn get_all_indices(&self) -> Vec<I>;
    fn generate_contiguous_multi_voxel_regions(
        &self,
        n_regions: usize,
    ) -> Result<(usize, Vec<Vec<(I, V)>>), CalcError>;
}

/// The different types of boundary conditions in a PDE system
/// One has to be careful, since the neumann condition is strictly speaking
/// not of the same type since its units are multiplied by 1/time compared to the others.
/// The Value variant is not a boundary condition in the traditional sense but
/// here used as the value which is present in another voxel.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum BoundaryCondition<ConcVecExtracellular> {
    Neumann(ConcVecExtracellular),
    Dirichlet(ConcVecExtracellular),
    Value(ConcVecExtracellular),
}

// TODO migrate to trait alias when stabilized
// pub trait Index = Ord + Hash + Eq + Clone + Send + Sync + Serialize + std::fmt::Debug;
pub trait Index: Ord + Hash + Eq + Clone + Send + Sync + Serialize + std::fmt::Debug {}
impl<T> Index for T where T: Ord + Hash + Eq + Clone + Send + Sync + Serialize + std::fmt::Debug {}

/* pub trait Concentration =
Sized + Add<Self, Output = Self> + Mul<f64, Output = Self> + Send + Sync + Zero;*/
pub trait Concentration:
    Sized + Add<Self, Output = Self> + Mul<f64, Output = Self> + Send + Sync + Zero
{
}
impl<T> Concentration for T where
    T: Sized + Add<Self, Output = Self> + Mul<f64, Output = Self> + Send + Sync + Zero
{
}

pub trait Voxel<I, Pos, Force>: Send + Sync + Clone + Serialize + for<'a> Deserialize<'a> {
    fn custom_force_on_cell(&self, _pos: &Pos) -> Option<Result<Force, CalcError>> {
        None
    }

    fn get_index(&self) -> I;

    // TODO these functions do NOT capture possible implementations accurately
    // In principle we should differentiate between
    //      - total concentrations everywhere in domain
    //          - some kind of additive/iterable multi-dimensional array
    //          - eg. in cartesian 2d: (n1,n2,m) array where n1,n2 are the number of sub-voxels and m the number of different concentrations
    //      - concentrations at a certain point
    //          - some kind of vector with entries corresponding to the individual concentrations
    //          - should be a slice of the total type
    //      - boundary conditions to adjacent voxels
    //          - some kind of multi-dimensional array with one dimension less than the total concentrations
    //          - should be a slice of the total type
    // In the future we hope to use https://doc.rust-lang.org/std/slice/struct.ArrayChunks.html

    // This is currently only a trait valid for n types of concentrations which are constant across the complete voxel
    // Functions related to diffusion and fluid dynamics of extracellular reactants/ligands
}

// TODO
// #[cfg(feature = "gradients")]
pub trait ExtracellularMechanics<
    Ind,
    Pos,
    ConcVec,
    ConcGradient,
    ConcTotal = ConcVec,
    ConcBoundary = ConcVec,
>: Send + Sync + Clone + Serialize + for<'a> Deserialize<'a>
{
    fn get_extracellular_at_point(&self, pos: &Pos) -> Result<ConcVec, SimulationError>;
    fn get_total_extracellular(&self) -> ConcTotal;
    #[cfg(feature = "gradients")]
    fn update_extracellular_gradient(
        &mut self,
        boundaries: &[(Ind, BoundaryCondition<ConcBoundary>)],
    ) -> Result<(), SimulationError>;
    #[cfg(feature = "gradients")]
    fn get_extracellular_gradient_at_point(
        &self,
        pos: &Pos,
    ) -> Result<ConcGradient, SimulationError>;
    fn set_total_extracellular(&mut self, concentration_total: &ConcTotal)
        -> Result<(), CalcError>;
    fn calculate_increment(
        &self,
        total_extracellular: &ConcTotal,
        point_sources: &[(Pos, ConcVec)],
        boundaries: &[(Ind, BoundaryCondition<ConcBoundary>)],
    ) -> Result<ConcTotal, CalcError>;
    fn boundary_condition_to_neighbor_voxel(
        &self,
        neighbor_index: &Ind,
    ) -> Result<BoundaryCondition<ConcBoundary>, IndexError>;
}
