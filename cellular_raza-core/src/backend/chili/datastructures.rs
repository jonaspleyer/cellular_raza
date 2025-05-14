use cellular_raza_concepts::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "tracing")]
use tracing::instrument;

use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;
use std::num::NonZeroUsize;

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
    pub subdomain_boxes: BTreeMap<I, Sb>,
}

/// Stores information related to a voxel of the physical simulation domain.
#[derive(Clone, Deserialize, Serialize)]
pub struct Voxel<C, A> {
    /// The index which is given when decomposing the domain and all indices are counted.
    pub plain_index: VoxelPlainIndex,
    /// Indices of neighboring voxels
    pub neighbors: BTreeSet<VoxelPlainIndex>,
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

/// Construct a new [SimulationRunner] from a given auxiliary storage and communicator object
#[allow(unused)]
#[cfg_attr(feature = "tracing", instrument(skip_all))]
pub fn construct_simulation_runner<D, S, C, A, Com, Sy, Ci>(
    domain: D,
    agents: Ci,
    n_subdomains: NonZeroUsize,
    init_aux_storage: impl Fn(&C) -> A,
) -> Result<
    SimulationRunner<D::SubDomainIndex, SubDomainBox<D::SubDomainIndex, S, C, A, Com, Sy>>,
    SimulationError,
>
where
    Ci: IntoIterator<Item = C>,
    D: Domain<C, S, Ci>,
    D::SubDomainIndex: Eq + PartialEq + core::hash::Hash + Clone + Ord,
    <S as SubDomain>::VoxelIndex: Eq + Hash + Ord + Clone,
    S: SortCells<C, VoxelIndex = <S as SubDomain>::VoxelIndex> + SubDomain,
    Sy: super::simulation_flow::FromMap<SubDomainPlainIndex>,
    Com: super::simulation_flow::FromMap<SubDomainPlainIndex>,
{
    #[cfg(feature = "tracing")]
    tracing::info!("Decomposing");
    let decomposed_domain = domain.decompose(n_subdomains, agents)?;

    #[cfg(feature = "tracing")]
    tracing::info!("Validating Map");
    if !validate_map(&decomposed_domain.neighbor_map) {
        return Err(SimulationError::IndexError(IndexError(format!(
            "Neighbor Map is invalid"
        ))));
    }

    #[cfg(feature = "tracing")]
    tracing::info!("Constructing Neighbor Map");
    let subdomain_index_to_subdomain_plain_index = decomposed_domain
        .index_subdomain_cells
        .iter()
        .enumerate()
        .map(|(i, (subdomain_index, _, _))| (subdomain_index.clone(), SubDomainPlainIndex(i)))
        .collect::<BTreeMap<_, _>>();
    let mut neighbor_map = decomposed_domain
        .neighbor_map
        .into_iter()
        .map(|(index, neighbors)| {
            (
                subdomain_index_to_subdomain_plain_index[&index],
                neighbors
                    .into_iter()
                    .filter_map(|sindex| {
                        if index != sindex {
                            Some(subdomain_index_to_subdomain_plain_index[&sindex])
                        } else {
                            None
                        }
                    })
                    .collect::<BTreeSet<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    #[cfg(feature = "tracing")]
    tracing::info!("Constructing Syncers and communicators");
    let mut syncers = Sy::from_map(&neighbor_map)?;
    let mut communicators = Com::from_map(&neighbor_map)?;
    let voxel_index_to_plain_index = decomposed_domain
        .index_subdomain_cells
        .iter()
        .map(|(_, subdomain, _)| subdomain.get_all_indices().into_iter())
        .flatten()
        .enumerate()
        .map(|(i, x)| (x, VoxelPlainIndex(i)))
        .collect::<BTreeMap<<S as SubDomain>::VoxelIndex, VoxelPlainIndex>>();
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

    #[cfg(feature = "tracing")]
    tracing::info!("Creating Subdomain Boxes");
    let subdomain_boxes = decomposed_domain
        .index_subdomain_cells
        .into_iter()
        .map(|(index, subdomain, cells)| {
            let subdomain_plain_index = subdomain_index_to_subdomain_plain_index[&index];
            let mut cells = cells.into_iter().map(|c| (c, None)).collect();
            let mut voxel_index_to_neighbor_plain_indices: BTreeMap<_, _> = subdomain
                .get_all_indices()
                .into_iter()
                .map(|voxel_index| {
                    (
                        voxel_index.clone(),
                        subdomain
                            .get_neighbor_voxel_indices(&voxel_index)
                            .into_iter()
                            .map(|neighbor_index| voxel_index_to_plain_index[&neighbor_index])
                            .collect::<BTreeSet<_>>(),
                    )
                })
                .collect();
            let voxels = subdomain.get_all_indices().into_iter().map(|voxel_index| {
                let plain_index = voxel_index_to_plain_index[&voxel_index];
                let neighbors = voxel_index_to_neighbor_plain_indices
                    .remove(&voxel_index)
                    .ok_or(super::IndexError(format!(
                        "cellular_raza::core::chili internal error in decomposition:\
                        please file a bug report!\
                        https://github.com/jonaspleyer/cellular_raza/issues/new?\
                        title=internal%20error%20during%20domain%20decomposition",
                    )))?;
                Ok((
                    plain_index,
                    Voxel {
                        plain_index,
                        neighbors,
                        cells: Vec::new(),
                        new_cells: Vec::new(),
                        id_counter: 0,
                        rng: rand_chacha::ChaCha8Rng::seed_from_u64(
                            decomposed_domain.rng_seed + plain_index.0 as u64,
                        ),
                    },
                ))
            });
            let syncer = syncers.remove(&subdomain_plain_index).ok_or(BoundaryError(
                "Index was not present in subdomain map".into(),
            ))?;
            use cellular_raza_concepts::BoundaryError;
            let communicator =
                communicators
                    .remove(&subdomain_plain_index)
                    .ok_or(BoundaryError(
                        "Index was not present in subdomain map".into(),
                    ))?;
            let mut subdomain_box = SubDomainBox {
                index: index.clone(),
                subdomain_plain_index,
                neighbors: neighbor_map
                    .remove(&subdomain_plain_index)
                    .unwrap()
                    .into_iter()
                    .collect(),
                subdomain,
                voxels: voxels.collect::<Result<_, SimulationError>>()?,
                voxel_index_to_plain_index: voxel_index_to_plain_index.clone(),
                plain_index_to_subdomain: plain_index_to_subdomain.clone(),
                communicator,
                syncer,
            };
            subdomain_box.insert_cells(&mut cells, &init_aux_storage)?;
            Ok((index, subdomain_box))
        })
        .collect::<Result<BTreeMap<_, _>, SimulationError>>()?;
    let simulation_runner = SimulationRunner { subdomain_boxes };
    Ok(simulation_runner)
}

/// Encapsulates a subdomain with cells and other simulation aspects.
pub struct SubDomainBox<I, S, C, A, Com, Sy = BarrierSync>
where
    S: SubDomain,
{
    #[allow(unused)]
    pub(crate) index: I,
    pub(crate) subdomain_plain_index: SubDomainPlainIndex,
    pub(crate) neighbors: BTreeSet<SubDomainPlainIndex>,
    pub(crate) subdomain: S,
    pub(crate) voxels: std::collections::BTreeMap<VoxelPlainIndex, Voxel<C, A>>,
    pub(crate) voxel_index_to_plain_index: BTreeMap<S::VoxelIndex, VoxelPlainIndex>,
    pub(crate) plain_index_to_subdomain:
        std::collections::BTreeMap<VoxelPlainIndex, SubDomainPlainIndex>,
    pub(crate) communicator: Com,
    pub(crate) syncer: Sy,
}

impl<I, S, C, A, Com, Sy> SubDomainBox<I, S, C, A, Com, Sy>
where
    S: SubDomain,
{
    /// Allows to sync between threads. In the most simplest
    /// case of [BarrierSync] syncing is done by a global barrier.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn sync(&mut self) -> Result<(), SimulationError>
    where
        Sy: SyncSubDomains,
    {
        self.syncer.sync()
    }

    /// Stores an error which has occurred and notifies other running threads to wind down.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn store_error(
        &mut self,
        maybe_error: Result<(), SimulationError>,
    ) -> Result<bool, SimulationError>
    where
        Sy: SyncSubDomains,
    {
        self.syncer.store_error(maybe_error)
    }

    // TODO this is not a boundary error!
    /// Allows insertion of cells into the subdomain.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn insert_cells(
        &mut self,
        new_cells: &mut Vec<(C, Option<A>)>,
        init_aux_storage: impl Fn(&C) -> A,
    ) -> Result<(), cellular_raza_concepts::BoundaryError>
    where
        <S as SubDomain>::VoxelIndex: Eq + Hash + Ord,
        S: SortCells<C, VoxelIndex = <S as SubDomain>::VoxelIndex>,
    {
        for (cell, aux_storage) in new_cells.drain(..) {
            let voxel_index = self.subdomain.get_voxel_index_of(&cell)?;
            let plain_index = self.voxel_index_to_plain_index[&voxel_index];
            let voxel = self.voxels.get_mut(&plain_index).ok_or(BoundaryError(
                "Could not find correct voxel for cell".to_owned(),
            ))?;
            let aux_storage = aux_storage.map_or(init_aux_storage(&cell), |x| x);
            voxel.cells.push((
                CellBox::new(voxel.plain_index, voxel.id_counter, cell, None),
                aux_storage,
            ));
            voxel.id_counter += 1;
        }
        Ok(())
    }

    /// Update all purely local functions
    ///
    /// Used to iterate over all cells in the current subdomain and running local functions which
    /// need no communication with other subdomains
    pub fn run_local_cell_funcs<Func, F>(
        &mut self,
        func: Func,
        next_time_point: &crate::time::NextTimePoint<F>,
    ) -> Result<(), super::SimulationError>
    where
        Func: Fn(
            &mut C,
            &mut A,
            F,
            &mut rand_chacha::ChaCha8Rng,
        ) -> Result<(), super::SimulationError>,
        F: Copy,
    {
        let dt = next_time_point.increment;
        for (_, voxel) in self.voxels.iter_mut() {
            for (cellbox, aux_storage) in voxel.cells.iter_mut() {
                func(&mut cellbox.cell, aux_storage, dt, &mut voxel.rng)?;
            }
        }
        Ok(())
    }

    /// TODO
    pub fn run_local_subdomain_funcs<Func, F>(
        &mut self,
        func: Func,
        next_time_point: &crate::time::NextTimePoint<F>,
    ) -> Result<(), super::SimulationError>
    where
        Func: Fn(
            &mut S,
            F,
            // &mut rand_chacha::ChaCha8Rng,
        ) -> Result<(), super::SimulationError>,
        F: Copy,
    {
        let dt = next_time_point.increment;
        func(&mut self.subdomain, dt)?;
        Ok(())
    }

    /// TODO
    pub fn run_cell_interaction_funcs<FuncG, G, FuncC, FuncV, FuncS, Ret>(
        &mut self,
        func_gather: FuncG,
        func_calculate: FuncC,
        func_send: FuncS,
    ) -> Result<(), super::SimulationError>
    where
        FuncG: Fn(&C) -> Result<G, super::SimulationError>,
        FuncC: Fn(&G, &C, &C, &mut A, &mut A) -> Result<(), super::SimulationError>,
        // TODO FuncS: Fn(&G, &C, &C, &mut A, &mut A) -> Result<(), super::SimulationError>,
        FuncV: Fn() -> Result<Ret, super::SimulationError>,
        FuncS:
            Fn(&VoxelPlainIndex, &VoxelPlainIndex, &G, usize) -> Result<(), super::SimulationError>,
    {
        let voxel_indices: Vec<_> = self.voxels.keys().map(|k| *k).collect();

        for voxel_index in voxel_indices {
            let n_cells = self.voxels[&voxel_index].cells.len();
            // Calculate Interactions internally
            for n in 0..n_cells {
                let vox = self.voxels.get_mut(&voxel_index).unwrap();

                let mut cells_mut = vox.cells.iter_mut();
                let (c1, aux1) = cells_mut.nth(n).unwrap();
                let gather = func_gather(&c1)?;
                for _ in n + 1..n_cells {
                    let (c2, aux2) = cells_mut.nth(0).unwrap();

                    func_calculate(&gather, c1, c2, aux1, aux2)?;
                }

                let neighbors = vox.neighbors.clone();
                for neighbor_index in &neighbors {
                    match self.voxels.get_mut(&neighbor_index) {
                        Some(vox) => {
                            let ret =
                                vox.run_cell_interaction_funcs_external(&gather, &func_calculate)?;
                        }
                        None => func_send(&voxel_index, &neighbor_index, &gather, n)?,
                    }
                }
            }
        }
        Ok(())
    }

    /// Save all subdomains with the given storage manager.
    #[cfg_attr(feature = "tracing", instrument(skip(self, storage_manager)))]
    pub fn save_subdomains<
        #[cfg(feature = "tracing")] F: core::fmt::Debug,
        #[cfg(not(feature = "tracing"))] F,
    >(
        &self,
        storage_manager: &mut crate::storage::StorageManager<SubDomainPlainIndex, S>,
        next_time_point: &crate::time::NextTimePoint<F>,
    ) -> Result<(), StorageError>
    where
        S: Clone + Serialize,
    {
        if let Some(crate::time::TimeEvent::PartialSave) = next_time_point.event {
            use crate::storage::StorageInterfaceStore;
            storage_manager.store_single_element(
                next_time_point.iteration as u64,
                &self.subdomain_plain_index,
                &self.subdomain,
            )?;
        }
        Ok(())
    }

    /// Stores all cells of the subdomain via the given [storage_manager](crate::storage)
    #[cfg_attr(feature = "tracing", instrument(skip(self, storage_manager)))]
    pub fn save_cells<
        #[cfg(feature = "tracing")] F: core::fmt::Debug,
        #[cfg(not(feature = "tracing"))] F,
    >(
        &self,
        storage_manager: &mut crate::storage::StorageManager<CellIdentifier, (CellBox<C>, A)>,
        next_time_point: &crate::time::NextTimePoint<F>,
    ) -> Result<(), StorageError>
    where
        A: Clone + Serialize,
        C: Clone + Serialize,
        CellBox<C>: cellular_raza_concepts::Id<Identifier = CellIdentifier>,
    {
        if let Some(crate::time::TimeEvent::PartialSave) = next_time_point.event {
            use crate::storage::StorageInterfaceStore;
            let cells = self
                .voxels
                .iter()
                .map(|(_, vox)| vox.cells.iter().map(|ca| (ca.0.ref_id(), ca)))
                .flatten();
            storage_manager.store_batch_elements(next_time_point.iteration as u64, cells)?;
        }
        Ok(())
    }
}
