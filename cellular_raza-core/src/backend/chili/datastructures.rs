use cellular_raza_concepts::errors::{BoundaryError, CalcError};
use serde::{Deserialize, Serialize};

use std::{hash::Hash, marker::PhantomData};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::aux_storage::*;
use super::concepts::*;
use super::simulation_flow::{BarrierSync, SyncSubDomains};

pub struct SimulationSupervisor<I, S, C, A, Sy>
where
    S: SubDomain<C>,
{
    subdomain_boxes: Vec<SubDomainBox<S, C, A, Sy>>,
    phandom_data: PhantomData<I>,
}

/// Stores information related to a voxel of the physical simulation domain.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Voxel<C, A> {
    /// Cells currently in the voxel
    pub cells: Vec<(C, A)>,
    /// New cells which are about to be included into this voxels cells.
    pub new_cells: Vec<C>,
    /// A counter to make sure that each Id of a cell is unique.
    pub id_counter: u64,
    /// A random number generator which is unique to this voxel and thus able
    /// to produce repeatable results even for parallelized simulations.
    pub rng: rand_chacha::ChaCha8Rng,
}

impl<I, S, C, A, Sy> From<DecomposedDomain<I, S, C>>
    for Result<SimulationSupervisor<I, S, C, A, Sy>, BoundaryError>
where
    S: SubDomain<C>,
    S::VoxelIndex: Eq + Ord,
    I: Eq + PartialEq + core::hash::Hash + Clone,
    A: Default,
    Sy: SyncSubDomains,
{
    // TODO this is not a BoundaryError
    ///
    fn from(
        decomposed_domain: DecomposedDomain<I, S, C>,
    ) -> Result<SimulationSupervisor<I, S, C, A, Sy>, BoundaryError> {
        let mut syncers = Sy::from_map(decomposed_domain.neighbor_map);

        let subdomain_boxes = decomposed_domain
            .index_subdomain_cells
            .into_iter()
            .map(|(index, subdomain, cells)| {
                let mut cells = cells.into_iter().map(|c| (c, None)).collect();
                let voxels = subdomain.get_all_indices().into_iter().map(|voxel_index| {
                    (
                        voxel_index,
                        Voxel {
                            cells: Vec::new(),
                            new_cells: Vec::new(),
                            id_counter: 0,
                            rng: rand_chacha::ChaCha8Rng::seed_from_u64(decomposed_domain.rng_seed),
                        },
                    )
                });
                let syncer = syncers.remove(&index).ok_or(BoundaryError {
                    message: "Index was not present in subdomain map".into(),
                })?;
                let mut subdomain_box = SubDomainBox {
                    subdomain,
                    voxels: voxels.collect(),
                    syncer,
                };
                subdomain_box.insert_cells(&mut cells)?;
                Ok(subdomain_box)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let simulation_supervisor = SimulationSupervisor {
            subdomain_boxes,
            phandom_data: PhantomData,
        };
        Ok(simulation_supervisor)
    }
}

#[doc(hidden)]
#[macro_export]
/// Given a collection of identifiers related to [concepts](cellular_raza_concepts) this will construct
/// a wrapper around a new [SimulationSupervisor] and implement the correct methods for the type.
///
/// Note: Some simulation aspects can be considered incompatible.
///
/// | Identifier | Concept | Depends on |
/// | --- | --- | --- |
/// | `Mechanics` | [Mechanics](cellular_raza_concepts::mechanics) | |
/// | `Cycle` | [Cycle](cellular_raza_concepts::cycle) | |
/// | `Interaction` | [Interaction](cellular_raza_concepts::interaction::Interaction) | |
/// | `CellularReactions` | [CellularReactions](cellular_raza_concepts::interaction::CellularReactions) | |
/// | `ExtracellularReactions` | [ExtracellularMechanics](cellular_raza_concepts::domain::ExtracellularMechanics) | |
/// | `Gradients` | [Gradients](cellular_raza_concepts::domain::ExtracellularMechanics) | `ExtracellularReactions` |
///
macro_rules! construct_supervisor(
    () => {};
);
#[doc(inline)]
pub use crate::construct_supervisor;

#[doc(hidden)]
#[macro_export]
/// A macro that checks if an identifier exists in a given range of identifiers and
/// inserts the given expression.
///
/// ```
/// # use cellular_raza_core::contains_ident;
/// contains_ident!(assert!(true), Mechanics, [Cycle, Interaction, Mechanics]);
///```
///
/// This will simply not insert the specified expression it will not fail or panic.
///```
/// # use cellular_raza_core::contains_ident;
/// contains_ident!(assert!(false), Mechanics, [Cycle, Interaction]);
/// ```
///
/// We need to specify at least one Identifier to match against
/// ```compile_fail
/// # use cellular_raza_core::contains_ident;
/// contains_ident!("Something", Mechanics, [])
/// ```
macro_rules! contains_ident(
    ($expression:expr, $id1:ident, [$($ids:ident),+]) => {
        $(
            cellular_raza_core_derive::identical!($expression, $id1, $ids);
        )+
    };
);
#[doc(inline)]
pub use crate::contains_ident;

/// Encapsulates a subdomain with cells and other simulation aspects.
pub struct SubDomainBox<S, C, A, Sy = BarrierSync>
where
    S: SubDomain<C>,
{
    pub(crate) subdomain: S,
    pub(crate) voxels: std::collections::BTreeMap<S::VoxelIndex, Voxel<C, A>>,
    pub(crate) syncer: Sy,
}

impl<S, C, A, Sy> SubDomainBox<S, C, A, Sy>
where
    S: SubDomain<C>,
{
    /// A subdomain can be initialized by specifying which cells are present how to sync between
    /// threads and the initial seed for our random number generator.
    pub fn initialize(subdomain: S, cells: Vec<C>, syncer: Sy, rng_seed: u64) -> Self
    where
        S::VoxelIndex: std::cmp::Eq + Hash + Ord,
        A: Default,
    {
        let voxel_indices = subdomain.get_all_indices();
        // TODO let voxels = subdomain.generate_all_voxels();
        let mut index_to_cells = cells
            .into_iter()
            .map(|cell| (subdomain.get_voxel_index_of(&cell).unwrap(), cell))
            .fold(
                std::collections::HashMap::new(),
                |mut acc, (index, cell)| {
                    let cells_in_voxel = acc.entry(index).or_insert(Vec::new());
                    cells_in_voxel.push((cell, A::default()));
                    acc
                },
            );
        let voxels = voxel_indices
            .into_iter()
            .map(|index| {
                let rng = ChaCha8Rng::seed_from_u64(rng_seed);
                let cells = index_to_cells.remove(&index).or(Some(Vec::new())).unwrap();
                (
                    index,
                    Voxel {
                        cells,
                        new_cells: Vec::new(),
                        id_counter: 0,
                        rng,
                    },
                )
            })
            .collect();
        Self {
            subdomain,
            voxels,
            syncer,
        }
    }

    /// Allows to sync between threads. In the most simplest case of [BarrierSync] syncing is done by a global barrier.
    pub fn sync(&mut self)
    where
        Sy: SyncSubDomains,
    {
        self.syncer.sync();
    }

    /// Applies boundary conditions to cells. For the future, we hope to be using previous and current position
    /// of cells rather than the cell itself.
    pub fn apply_boundary(&mut self) -> Result<(), BoundaryError> {
        self.voxels
            .iter_mut()
            .map(|(_, voxel)| voxel.cells.iter_mut())
            .flatten()
            .map(|(cell, _)| self.subdomain.apply_boundary(cell))
            .collect::<Result<(), BoundaryError>>()
    }

    // TODO this is not a boundary error!
    /// Allows insertion of cells into the subdomain.
    pub fn insert_cells(&mut self, new_cells: &mut Vec<(C, Option<A>)>) -> Result<(), BoundaryError>
    where
        S::VoxelIndex: Ord,
        A: Default,
    {
        for cell in new_cells.drain(..) {
            let voxel_index = self.subdomain.get_voxel_index_of(&cell.0)?;
            self.voxels
                .get_mut(&voxel_index)
                .ok_or(BoundaryError {
                    message: "Could not find correct voxel for cell".to_owned(),
                })?
                .cells
                .push((cell.0, cell.1.map_or(A::default(), |x| x)));
        }
        Ok(())
    }

    /// Advances the cycle of a cell by a small time increment `dt`.
    pub fn update_cycle(&mut self, dt: f64) -> Result<(), CalcError>
    where
        C: cellular_raza_concepts::cycle::Cycle<C>,
        A: UpdateCycle,
    {
        self.voxels.iter_mut().for_each(|(_, voxel)| {
            voxel.cells.iter_mut().for_each(|(cell, aux_storage)| {
                if let Some(event) = C::update_cycle(&mut voxel.rng, &dt, cell) {
                    aux_storage.add_cycle_event(event);
                }
            })
        });
        Ok(())
    }
}
