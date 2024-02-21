use cellular_raza_concepts::domain_new::{DecomposedDomain, SubDomain};
use cellular_raza_concepts::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "tracing")]
use tracing::instrument;

use std::collections::HashMap;
use std::hash::Hash;

use rand::SeedableRng;

use super::aux_storage::*;
use super::errors::*;
use super::simulation_flow::*;

use super::{CellIdentifier, SubDomainPlainIndex, VoxelPlainIndex};

/// Intermediate object which gets consumed once the simulation is run
///
/// This Setup contains structural information needed to run a simulation.
/// In the future, we hope to change the types stored in this object to
/// simple iterators and non-allocating types in general.
pub struct SimulationRunner<I, Sb> {
    // TODO make this private
    /// One [SubDomainBox] represents one single thread over which we are parallelizing
    /// our simulation.
    pub subdomain_boxes: HashMap<I, Sb>,
}

/// Stores information related to a voxel of the physical simulation domain.
#[derive(Clone, Deserialize, Serialize)]
pub struct Voxel<C, A> {
    /// The index which is given when decomposing the domain and all indices are counted.
    pub plain_index: VoxelPlainIndex,
    /// Indices of neighboring voxels
    pub neighbors: Vec<VoxelPlainIndex>,
    /// Cells currently in the voxel
    pub cells: Vec<(CellBox<C>, A)>,
    /// New cells which are about to be included into this voxels cells.
    pub new_cells: Vec<(C, Option<CellIdentifier>)>,
    /// A counter to make sure that each Id of a cell is unique.
    pub id_counter: u64,
    /// A random number generator which is unique to this voxel and thus able
    /// to produce repeatable results even for parallelized simulations.
    pub rng: rand_chacha::ChaCha8Rng,
}

impl<C, A> Voxel<C, A> {
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub(crate) fn update_cell_cycle_3<
        #[cfg(feature = "tracing")] Float: core::fmt::Debug,
        #[cfg(not(feature = "tracing"))] Float,
    >(
        &mut self,
        dt: &Float,
    ) -> Result<(), SimulationError>
    where
        C: cellular_raza_concepts::Cycle<C, Float>
            + cellular_raza_concepts::domain_new::Id<Identifier = CellIdentifier>,
        A: UpdateCycle + Default,
    {
        // Update the cell individual cells
        self.cells
            .iter_mut()
            .map(|(cbox, aux_storage)| {
                // Check for cycle events and do update if necessary
                let mut remaining_events = Vec::new();
                for event in aux_storage.drain_cycle_events() {
                    match event {
                        CycleEvent::Division => {
                            let new_cell = C::divide(&mut self.rng, &mut cbox.cell)?;
                            self.new_cells.push((new_cell, Some(cbox.get_id())));
                        }
                        CycleEvent::Remove => remaining_events.push(event),
                        CycleEvent::PhasedDeath => {
                            remaining_events.push(event);
                        }
                    };
                }
                aux_storage.set_cycle_events(remaining_events);
                // Update the cell cycle
                if aux_storage
                    .get_cycle_events()
                    .contains(&CycleEvent::PhasedDeath)
                {
                    match C::update_conditional_phased_death(&mut self.rng, dt, &mut cbox.cell)? {
                        true => aux_storage.add_cycle_event(CycleEvent::Remove),
                        false => (),
                    }
                } else {
                    match C::update_cycle(&mut self.rng, dt, &mut cbox.cell) {
                        Some(event) => aux_storage.add_cycle_event(event),
                        None => (),
                    }
                }
                Ok(())
            })
            .collect::<Result<(), SimulationError>>()?;

        // Remove cells which are flagged for death
        self.cells.retain(|(_, aux_storage)| {
            !aux_storage.get_cycle_events().contains(&CycleEvent::Remove)
        });

        // Include new cells
        self.cells
            .extend(self.new_cells.drain(..).map(|(cell, parent_id)| {
                self.id_counter += 1;
                (
                    CellBox::new(self.plain_index, self.id_counter, cell, parent_id),
                    A::default(),
                )
            }));
        Ok(())
    }
}

impl<I, S, C, A, Com, Sy> From<DecomposedDomain<I, S, C>>
    for SimulationRunner<I, SubDomainBox<I, S, C, A, Com, Sy>>
where
    S: SubDomain<C>,
    S::VoxelIndex: Eq + Hash + Ord + Clone,
    I: Eq + PartialEq + core::hash::Hash + Clone + Ord,
    A: Default,
    Sy: super::simulation_flow::FromMap<SubDomainPlainIndex>,
    Com: super::simulation_flow::FromMap<SubDomainPlainIndex>,
{
    // TODO this is not a BoundaryError
    ///
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    fn from(
        decomposed_domain: DecomposedDomain<I, S, C>,
    ) -> SimulationRunner<I, SubDomainBox<I, S, C, A, Com, Sy>> {
        // TODO do not unwrap
        if !validate_map(&decomposed_domain.neighbor_map) {
            panic!("Map not valid!");
        }
        let subdomain_index_to_subdomain_plain_index = decomposed_domain
            .index_subdomain_cells
            .iter()
            .enumerate()
            .map(|(i, (subdomain_index, _, _))| (subdomain_index.clone(), SubDomainPlainIndex(i)))
            .collect::<HashMap<_, _>>();
        let neighbor_map = decomposed_domain
            .neighbor_map
            .into_iter()
            .map(|(index, neighbors)| {
                (
                    subdomain_index_to_subdomain_plain_index[&index],
                    neighbors
                        .into_iter()
                        .map(|index| subdomain_index_to_subdomain_plain_index[&index])
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<HashMap<_, _>>();
        let mut syncers = Sy::from_map(&neighbor_map).unwrap();
        let mut communicators = Com::from_map(&neighbor_map).unwrap();
        let voxel_index_to_plain_index = decomposed_domain
            .index_subdomain_cells
            .iter()
            .map(|(_, subdomain, _)| subdomain.get_all_indices().into_iter())
            .flatten()
            .enumerate()
            .map(|(i, x)| (x, VoxelPlainIndex(i)))
            .collect::<HashMap<S::VoxelIndex, VoxelPlainIndex>>();
        let plain_index_to_subdomain: std::collections::BTreeMap<_, _> = decomposed_domain
            .index_subdomain_cells
            .iter()
            .enumerate()
            .map(|(subdomain_index, (_, subdomain, _))| {
                subdomain
                    .get_all_indices()
                    .into_iter()
                    .map(move |index| (subdomain_index, index))
            })
            .flatten()
            .map(|(subdomain_index, voxel_index)| {
                (
                    voxel_index_to_plain_index[&voxel_index],
                    SubDomainPlainIndex(subdomain_index),
                )
            })
            .collect();

        let subdomain_boxes = decomposed_domain
            .index_subdomain_cells
            .into_iter()
            .map(|(index, subdomain, cells)| {
                let subdomain_plain_index = subdomain_index_to_subdomain_plain_index[&index];
                let mut cells = cells.into_iter().map(|c| (c, None)).collect();
                let mut voxel_index_to_neighbor_plain_indices: HashMap<_, _> = subdomain
                    .get_all_indices()
                    .into_iter()
                    .map(|voxel_index| {
                        (
                            voxel_index.clone(),
                            subdomain
                                .get_neighbor_voxel_indices(&voxel_index)
                                .into_iter()
                                .map(|neighbor_index| voxel_index_to_plain_index[&neighbor_index])
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();
                let voxels = subdomain.get_all_indices().into_iter().map(|voxel_index| {
                    let plain_index = voxel_index_to_plain_index[&voxel_index];
                    let neighbors = voxel_index_to_neighbor_plain_indices
                        .remove(&voxel_index)
                        .unwrap();
                    (
                        plain_index,
                        Voxel {
                            plain_index,
                            neighbors,
                            cells: Vec::new(),
                            new_cells: Vec::new(),
                            id_counter: 0,
                            rng: rand_chacha::ChaCha8Rng::seed_from_u64(decomposed_domain.rng_seed),
                        },
                    )
                });
                let syncer = syncers.remove(&subdomain_plain_index).ok_or(BoundaryError(
                    "Index was not present in subdomain map".into(),
                ))?;
                let communicator =
                    communicators
                        .remove(&subdomain_plain_index)
                        .ok_or(BoundaryError(
                            "Index was not present in subdomain map".into(),
                        ))?;
                let mut subdomain_box = SubDomainBox {
                    _index: index.clone(),
                    subdomain,
                    voxels: voxels.collect(),
                    voxel_index_to_plain_index: voxel_index_to_plain_index.clone(),
                    plain_index_to_subdomain: plain_index_to_subdomain.clone(),
                    communicator,
                    syncer,
                };
                subdomain_box.insert_cells(&mut cells)?;
                Ok((index, subdomain_box))
            })
            .collect::<Result<HashMap<_, _>, BoundaryError>>()
            .unwrap();
        let simulatino_runner = SimulationRunner { subdomain_boxes };
        simulatino_runner
    }
}

/// Encapsulates a subdomain with cells and other simulation aspects.
pub struct SubDomainBox<I, S, C, A, Com, Sy = BarrierSync>
where
    S: SubDomain<C>,
{
    pub(crate) _index: I,
    pub(crate) subdomain: S,
    pub(crate) voxels: std::collections::BTreeMap<VoxelPlainIndex, Voxel<C, A>>,
    pub(crate) voxel_index_to_plain_index:
        std::collections::HashMap<S::VoxelIndex, VoxelPlainIndex>,
    pub(crate) plain_index_to_subdomain:
        std::collections::BTreeMap<VoxelPlainIndex, SubDomainPlainIndex>,
    pub(crate) communicator: Com,
    pub(crate) syncer: Sy,
}

impl<I, S, C, A, Com, Sy> SubDomainBox<I, S, C, A, Com, Sy>
where
    S: SubDomain<C>,
{
    /// Allows to sync between threads. In the most simplest case of [BarrierSync] syncing is done by a global barrier.
    pub fn sync(&mut self)
    where
        Sy: SyncSubDomains,
    {
        self.syncer.sync();
    }

    // TODO this is not a boundary error!
    /// Allows insertion of cells into the subdomain.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn insert_cells(&mut self, new_cells: &mut Vec<(C, Option<A>)>) -> Result<(), BoundaryError>
    where
        S::VoxelIndex: Eq + Hash + Ord,
        A: Default,
    {
        for (cell, aux_storage) in new_cells.drain(..) {
            let voxel_index = self.subdomain.get_voxel_index_of(&cell)?;
            let plain_index = self.voxel_index_to_plain_index[&voxel_index];
            let voxel = self.voxels.get_mut(&plain_index).ok_or(BoundaryError(
                "Could not find correct voxel for cell".to_owned(),
            ))?;
            voxel.cells.push((
                CellBox::new(voxel.plain_index, voxel.id_counter, cell, None),
                aux_storage.map_or(A::default(), |x| x),
            ));
        }
        Ok(())
    }

    /// Advances the cycle of a cell by a small time increment `dt`.
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub fn update_cycle<
        #[cfg(feature = "tracing")] Float: core::fmt::Debug,
        #[cfg(not(feature = "tracing"))] Float,
    >(
        &mut self,
        dt: Float,
    ) -> Result<(), CalcError>
    where
        C: cellular_raza_concepts::Cycle<C, Float>,
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

impl<I, S, C, A, Com, Sy> SubDomainBox<I, S, C, A, Com, Sy>
where
    S: SubDomain<C>,
{
    /// Separate function to update the cell cycle
    ///
    /// Instead of running one big update function for all local rules, we have to treat this cell
    /// cycle differently since new cells could be generated and thus have consequences for other
    /// update steps as well.
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub fn update_cell_cycle_3<
        #[cfg(feature = "tracing")] F: core::fmt::Debug,
        #[cfg(not(feature = "tracing"))] F,
    >(
        &mut self,
        dt: &F,
    ) -> Result<(), SimulationError>
    where
        C: cellular_raza_concepts::Cycle<C, F>
            + cellular_raza_concepts::domain_new::Id<Identifier = CellIdentifier>,
        A: UpdateCycle + Default,
    {
        self.voxels
            .iter_mut()
            .map(|(_, vox)| vox.update_cell_cycle_3(dt))
            .collect::<Result<(), SimulationError>>()?;
        Ok(())
    }

    /// Save all voxels (containing all cells) with the given storage manager.
    #[cfg_attr(feature = "tracing", instrument(skip(self, storage_manager)))]
    pub fn save_voxels<
        #[cfg(feature = "tracing")] F: core::fmt::Debug,
        #[cfg(not(feature = "tracing"))] F,
    >(
        &self,
        storage_manager: &crate::storage::StorageManager<&VoxelPlainIndex, &Voxel<C, A>>,
        next_time_point: &crate::time::NextTimePoint<F>,
    ) -> Result<(), StorageError>
    where
        Voxel<C, A>: Serialize,
    {
        if let Some(crate::time::TimeEvent::PartialSave) = next_time_point.event {
            let voxels = self.voxels.iter().collect::<Vec<_>>();
            use crate::storage::StorageInterface;
            storage_manager.store_batch_elements(next_time_point.iteration as u64, &voxels)?;
        }
        Ok(())
    }

    /// Stores all cells of the subdomain via the given [storage_manager}(crate::storage)
    #[cfg_attr(feature = "tracing", instrument(skip(self, storage_manager)))]
    pub fn save_cells<
        #[cfg(feature = "tracing")] F: core::fmt::Debug,
        #[cfg(not(feature = "tracing"))] F,
    >(
        &self,
        storage_manager: &crate::storage::StorageManager<&CellIdentifier, (&CellBox<C>, &A)>,
        next_time_point: &crate::time::NextTimePoint<F>,
    ) -> Result<(), StorageError>
    where
        A: Serialize,
        C: Serialize,
        CellBox<C>: cellular_raza_concepts::domain_new::Id<Identifier = CellIdentifier>,
    {
        if let Some(crate::time::TimeEvent::PartialSave) = next_time_point.event {
            use crate::storage::StorageInterface;
            use cellular_raza_concepts::domain_new::Id;
            let cells = self
                .voxels
                .iter()
                .map(|(_, vox)| vox.cells.iter())
                .flatten()
                .map(|(c, a)| (c.ref_id(), (c, a)))
                .collect::<Vec<_>>();
            storage_manager.store_batch_elements(next_time_point.iteration as u64, &cells)?;
        }
        Ok(())
    }
}
