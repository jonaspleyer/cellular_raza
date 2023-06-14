use crate::cell::CellAgentBox;
use crate::errors::*;

use std::marker::{Send, Sync};

use core::cmp::Eq;
use core::hash::Hash;
use std::ops::{Add, Mul};

use num::Zero;
use serde::{Deserialize, Serialize};

use super::cycle::CycleEvent;

/// Describes the physical simulation domain.
///
/// This trait is responsible for the overall phyiscal setup of our simulation.
/// Simple domains can be thought of as rectangular cuboids in 2D or 3D and such examples
/// are being implemented in [domains](https://docs.rs/cellular_raza-building-blocks::domains).
/// Most of the functions you will see are required by the backend to make sense of the whole simulation domain and
/// send cells to the correct subdomains and give correct boundary conditions.
/// [cellular_raza](https://docs.rs/cellular_raza) uses [domain decomposition](https://wikipedia.org/wiki/Domain_decomposition_methods)
/// to parallelize over different regions of space and thus efficiently utilize hardware resources
/// although the exact details do depend on the [backend](https://docs.rs/cellular_raza-core/backend/).
/// The simulation domain is split into many voxels which interact only with their next neighbors.
/// By increasing the size of these voxels, we can allow for longer interaction ranges or vice versa.
pub trait Domain<C, I, V>: Send + Sync + Serialize + for<'a> Deserialize<'a> {
    /// Applies boundary conditions to a cell in order to keep it inside the simulation.
    /// For the future, we aim to apply boundary conditions to the position of the cell rather than itself.
    /// In addition, we would like to be able to invoke events such as [Remove](super::cycle::CycleEvent::Remove) to maximize flexibility.
    fn apply_boundary(&self, cell: &mut C) -> Result<(), BoundaryError>;

    /// Retrieves the neighboring voxels of the one specified.
    fn get_neighbor_voxel_indices(&self, index: &I) -> Vec<I>;

    /// Provided a cell, gives the corresponding Index and thus which voxel to sort into.
    fn get_voxel_index(&self, cell: &C) -> I;

    /// Get all indices that are present in the simulation. Required for initial configuration of the simulation domain.
    fn get_all_indices(&self) -> Vec<I>;
    // TODO rename this function and generate SubDomains which then hold a number of voxels.
    // These subdomains should also be responsible to integrate extracellular mechanics and so on.
    // This is already partly realized by MultivoxelContainers in the domain_decomposition module of the cpu_os_threads backend.
    /// Allows the backend to split the domain into continuous regions which contain voxels.
    /// These regions can then be used for parallelization.
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
    /// Neumann boundary conditions apply a value to the derivative at the corresponding boundary.
    Neumann(ConcVecExtracellular),
    /// Dirichlet conditions fix the value of concentration at the boundary.
    Dirichlet(ConcVecExtracellular),
    /// This boundary condition fixes the value at boundary. Although in principle exactly the same as [BoundaryCondition::Dirichlet],
    /// this value is not provided by the user but rather the boundary condition of another simulation subdomain.
    Value(ConcVecExtracellular),
}

// TODO migrate to trait alias when stabilized
// pub trait Index = Ord + Hash + Eq + Clone + Send + Sync + Serialize + std::fmt::Debug;
/// Summarizes traits required for an [Index] of a [Domain] to work.
pub trait Index: Ord + Hash + Eq + Clone + Send + Sync + Serialize + std::fmt::Debug {}
impl<T> Index for T where T: Ord + Hash + Eq + Clone + Send + Sync + Serialize + std::fmt::Debug {}

/* pub trait Concentration =
Sized + Add<Self, Output = Self> + Mul<f64, Output = Self> + Send + Sync + Zero;*/
/// Preliminary traits to model external [Concentration] in respective [Voxel].
pub trait Concentration:
    Sized + Add<Self, Output = Self> + Mul<f64, Output = Self> + Send + Sync + Zero
{
}
impl<T> Concentration for T where
    T: Sized + Add<Self, Output = Self> + Mul<f64, Output = Self> + Send + Sync + Zero
{
}

/// The [Domain] trait generalizes over the [Voxel] generic parameter which 
pub trait Voxel<Ind, Pos, Vel, Force>:
    Send + Sync + Clone + Serialize + for<'a> Deserialize<'a>
{
    /// Voxels can exert custom forces on cells. In the future this function will probably be part of a different Trait.
    fn custom_force_on_cell(&self, _pos: &Pos, _vel: &Vel) -> Option<Result<Force, CalcError>> {
        None
    }

    /// Gets the Index of the voxel.
    fn get_index(&self) -> Ind;
}

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

// TODO rework this trait. We do not want to make it dependent on the voxel index. But it may be dependent on spatial representation such as Position type.
/// First approach on how to generalize over extracellular mechanics.
/// Future versions will not depend on the [Voxel] [Index] but be more general.
pub trait ExtracellularMechanics<
    Ind,
    Pos,
    ConcVec,
    ConcGradient,
    ConcTotal = ConcVec,
    ConcBoundary = ConcVec,
>: Send + Sync + Clone + Serialize + for<'a> Deserialize<'a>
{
    /// Obtain the extracellular concentration at a specified point.
    fn get_extracellular_at_point(&self, pos: &Pos) -> Result<ConcVec, RequestError>;

    /// Obtain every concentration in the current voxel. This function is only relevant for the [backend](https://docs.rs/cellular_raza-core/backend/).
    fn get_total_extracellular(&self) -> ConcTotal;

    // TODO formulate additional trait which does extracellular gradients mechanics and can be linked to this trait
    /// Update function to calculate the gradient of concentrations in this voxel.
    #[cfg(feature = "gradients")]
    fn update_extracellular_gradient(
        &mut self,
        boundaries: &[(Ind, BoundaryCondition<ConcBoundary>)],
    ) -> Result<(), CalcError>;

    // TODO formulate additional trait which does extracellular gradients mechanics and can be linked to this trait
    /// Obtain the gradient at a certain point.
    #[cfg(feature = "gradients")]
    fn get_extracellular_gradient_at_point(&self, pos: &Pos) -> Result<ConcGradient, RequestError>;

    /// Simple setter function to specify concentrations after backend has updated values.
    fn set_total_extracellular(&mut self, concentration_total: &ConcTotal)
        -> Result<(), CalcError>;

    /// Calculates the time-derivative of the function that increments the concentrations.
    fn calculate_increment(
        &self,
        total_extracellular: &ConcTotal,
        point_sources: &[(Pos, ConcVec)],
        boundaries: &[(Ind, BoundaryCondition<ConcBoundary>)],
    ) -> Result<ConcTotal, CalcError>;

    /// Gets the boundary to the specified neighboring voxel.
    fn boundary_condition_to_neighbor_voxel(
        &self,
        neighbor_index: &Ind,
    ) -> Result<BoundaryCondition<ConcBoundary>, IndexError>;
}

/// An external controller which can see all of the simulation domain's cells and perform modifications on individual cells.
/// This controller is only useful when describing systems that are controlled from the outside and not subject to local interactions.
/// This trait is not finalized yet.
pub trait Controller<C, O> {
    /// This function views a part of the simulation domain and retrieves information about the cells contained in it.
    /// Afterwards, this measurement is stored and then a collection of measurements is provided to the [adjust](Controller::adjust) function.
    fn measure<'a, I>(&self, cells: I) -> Result<O, CalcError>
    where
        C: 'a + Serialize + for<'b> Deserialize<'b>,
        I: IntoIterator<Item = &'a CellAgentBox<C>> + Clone;

    /// Function that operates on cells given an iterator over measurements. It modifies cellular properties and can invoke [CycleEvenets](super::cycle::CycleEvent).
    fn adjust<'a, 'b, I, J>(&mut self, measurements: I, cells: J) -> Result<(), ControllerError>
    where
        O: 'a,
        C: 'b + Serialize + for<'c> Deserialize<'c>,
        I: Iterator<Item = &'a O>,
        J: Iterator<Item = (&'b mut CellAgentBox<C>, &'b mut Vec<CycleEvent>)>;
}

impl<C> Controller<C, ()> for () {
    fn measure<'a, I>(&self, _cells: I) -> Result<(), CalcError>
    where
        C: 'a + Serialize + for<'b> Deserialize<'b>,
        I: IntoIterator<Item = &'a CellAgentBox<C>> + Clone,
    {
        Ok(())
    }

    #[allow(unused)]
    fn adjust<'a, 'b, I, J>(&mut self, measurements: I, cells: J) -> Result<(), ControllerError>
    where
        (): 'a,
        C: 'b + Serialize + for<'c> Deserialize<'c>,
        I: Iterator<Item = &'a ()>,
        J: Iterator<Item = (&'b mut CellAgentBox<C>, &'b mut Vec<CycleEvent>)>,
    {
        Ok(())
    }
}
