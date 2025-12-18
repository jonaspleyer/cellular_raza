use crate::domain::VoxelPlainIndex;
use crate::errors::{CalcError, RngError};
use crate::interaction::*;
use crate::mechanics::{Mechanics, Position, Velocity};

use serde::{Deserialize, Serialize};

// TODO move this module to cpu_os_threads backend except for traits

/// Unique identifier which is given to every cell in the simulation
///
/// The identifier is comprised of the [VoxelPlainIndex] in which the cell was first spawned.
/// This can be due to initial setup or due to other methods such as division in a cell cycle.
/// The second parameter is a counter which is unique for each voxel.
/// This ensures that each cell obtains a unique identifier over the course of the simulation.
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
#[derive(Clone, Copy, Debug, Deserialize, Hash, PartialEq, Eq, Ord, PartialOrd, Serialize)]
pub enum CellIdentifier {
    /// Produced from a division process
    Division(VoxelPlainIndex, u64),
    /// Initially placed inside the simulation
    Initial(usize),
    /// Inserted manually by other processes
    Inserted(VoxelPlainIndex, u64),
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl CellIdentifier {
    /// Constructs a new [CellIdentifier::Division]
    #[new]
    pub fn new(voxel_plain_index: VoxelPlainIndex, counter: u64) -> Self {
        CellIdentifier::Division(voxel_plain_index, counter)
    }

    /// Construct a new [CellIdentifier::Initial]
    #[staticmethod]
    pub fn new_initial(index: usize) -> Self {
        CellIdentifier::Initial(index)
    }

    /// Construct a new [CellIdentifier::Inserted]
    #[staticmethod]
    pub fn new_inserted(voxel_plain_index: VoxelPlainIndex, counter: u64) -> Self {
        Self::Inserted(voxel_plain_index, counter)
    }

    /// Returns an identical clone of the identifier
    pub fn __deepcopy__(&self, _memo: pyo3::Bound<pyo3::types::PyDict>) -> Self {
        *self
    }

    /// Returns an identical clone of the identifier
    pub fn copy(&self) -> Self {
        *self
    }

    /// Returns an identical clone of the identifier
    pub fn __copy__(&self) -> Self {
        *self
    }

    /// Formats the CellIdentifier
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    /// Performs the `==` operation.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.eq(other)
    }

    /// Calculates a hash value of type `u64`
    pub fn __hash__(&self) -> u64 {
        use core::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    ///  Performs the `<` operation
    pub fn __lt__(&self, other: &Self) -> bool {
        self.lt(other)
    }

    /// Implementes the `__getitem__` method. Since the [CellIdentifier] is built like a list this
    /// only works for the entires 0 and 1 and will yield an error otherwise
    pub fn __getitem__<'py>(
        &self,
        py: pyo3::Python<'py>,
        key: usize,
    ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        use pyo3::IntoPyObject;
        let (key0, key1) = match self {
            CellIdentifier::Initial(key0) => (*key0, None),
            CellIdentifier::Division(key0, key1) => (key0.0, Some(*key1)),
            CellIdentifier::Inserted(key0, key1) => (key0.0, Some(*key1)),
        };
        if key == 0 {
            Ok(key0.into_pyobject(py)?.into_any())
        } else if key == 1 {
            Ok(key1.into_pyobject(py)?.into_any())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "CellIdentifier can only be indexed at 0 and 1",
            ))
        }
    }
}

/// Specifies how to retrieve a unique identifier of an object.
pub trait Id {
    /// The identifier type is usually chosen to be completely unique and repeatable across
    /// different simulations.
    type Identifier;

    /// Retrieves the Identifier from the object.
    fn get_id(&self) -> Self::Identifier;
    /// Returns a reference to the id of the object.
    fn ref_id(&self) -> &Self::Identifier;
}

/// Wrapper around the user-defined CellAgent
///
/// This wrapper serves to provide a unique identifier and the option to specify
/// the parent of the current cell.
#[derive(Clone, Deserialize, Serialize)]
pub struct CellBox<C> {
    /// The identifier is composed of two values, one for the voxel index in which the
    /// object was created and another one which counts how many elements have already
    /// been created there.
    pub identifier: CellIdentifier,
    /// Identifier of the parent cell if this cell was created by cell-division
    pub parent: Option<CellIdentifier>,
    /// The cell which is encapsulated by this box.
    pub cell: C,
}

impl<Cel> Id for CellBox<Cel> {
    type Identifier = CellIdentifier;

    fn get_id(&self) -> CellIdentifier {
        self.identifier
    }

    fn ref_id(&self) -> &CellIdentifier {
        &self.identifier
    }
}

impl<Cel> CellBox<Cel> {
    /// Simple method to retrieve the [CellularIdentifier] of the parent cell if existing.
    pub fn get_parent_id(&self) -> Option<CellIdentifier> {
        self.parent
    }
}

impl<Inf, A> InteractionInformation<Inf> for CellBox<A>
where
    A: InteractionInformation<Inf>,
{
    fn get_interaction_information(&self) -> Inf {
        self.cell.get_interaction_information()
    }
}

// Auto-implement traits for CellAgentBox which where also implemented for Agent
impl<Pos, Vel, For, Inf, A> Interaction<Pos, Vel, For, Inf> for CellBox<A>
where
    A: Interaction<Pos, Vel, For, Inf> + Serialize + for<'a> Deserialize<'a>,
{
    fn calculate_force_between(
        &self,
        own_pos: &Pos,
        own_vel: &Vel,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_information: &Inf,
    ) -> Result<(For, For), CalcError> {
        self.cell
            .calculate_force_between(own_pos, own_vel, ext_pos, ext_vel, ext_information)
    }
}

impl<A, Pos> Position<Pos> for CellBox<A>
where
    A: Position<Pos>,
{
    fn pos(&self) -> Pos {
        self.cell.pos()
    }

    fn set_pos(&mut self, pos: &Pos) {
        self.cell.set_pos(pos)
    }
}

impl<A, Vel> Velocity<Vel> for CellBox<A>
where
    A: Velocity<Vel>,
{
    fn velocity(&self) -> Vel {
        self.cell.velocity()
    }

    fn set_velocity(&mut self, velocity: &Vel) {
        self.cell.set_velocity(velocity)
    }
}

impl<Pos, Vel, For, Float, A> Mechanics<Pos, Vel, For, Float> for CellBox<A>
where
    A: Mechanics<Pos, Vel, For, Float>,
{
    fn get_random_contribution(
        &self,
        rng: &mut rand_chacha::ChaCha8Rng,
        dt: Float,
    ) -> Result<(Pos, Vel), RngError> {
        self.cell.get_random_contribution(rng, dt)
    }

    fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError> {
        self.cell.calculate_increment(force)
    }
}

impl<C> core::ops::Deref for CellBox<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.cell
    }
}

impl<C> core::ops::DerefMut for CellBox<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cell
    }
}

impl<C> CellBox<C> {
    /// Create a new [CellBox] for a cell present initially in the simulation.
    pub fn new_initial(n_cell: usize, cell: C) -> CellBox<C> {
        CellBox::<C> {
            identifier: CellIdentifier::Initial(n_cell),
            parent: None,
            cell,
        }
    }

    /// Create a new [CellBox] at a specific voxel with a voxel-unique number
    /// of cells that has already been created at this position.
    pub fn new(
        voxel_index: VoxelPlainIndex,
        n_cell: u64,
        cell: C,
        parent: Option<CellIdentifier>,
    ) -> CellBox<C> {
        CellBox::<C> {
            identifier: CellIdentifier::Division(voxel_index, n_cell),
            parent,
            cell,
        }
    }
}

#[doc(inline)]
pub use cellular_raza_concepts_derive::CellAgent;
