use cellular_raza_concepts::domain_new::{DecomposedDomain, SubDomain};
use cellular_raza_concepts::*;
use num::FromPrimitive;
use serde::{Deserialize, Serialize};

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
    pub(crate) fn calculate_force_between_cells_internally<
        Pos,
        Vel,
        For,
        Float,
        Inf,
        const N: usize,
    >(
        &mut self,
    ) -> Result<(), CalcError>
    where
        C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>,
        A: UpdateMechanics<Pos, Vel, For, Float, N>,
        For: Clone + core::ops::Mul<Float, Output = For> + core::ops::Neg<Output = For>,
        Float: num::Float,
    {
        let one_half: Float = Float::one() / (Float::one() + Float::one());

        for n in 0..self.cells.len() {
            for m in n + 1..self.cells.len() {
                let mut cells_mut = self.cells.iter_mut();
                let (c1, aux1) = cells_mut.nth(n).unwrap();
                let (c2, aux2) = cells_mut.nth(m - n - 1).unwrap();

                let p1 = c1.pos();
                let v1 = c1.velocity();
                let i1 = c1.get_interaction_information();

                let p2 = c2.pos();
                let v2 = c2.velocity();
                let i2 = c2.get_interaction_information();

                if let Some(force_result) = c1.calculate_force_between(&p1, &v1, &p2, &v2, &i2) {
                    let force = force_result?;
                    aux1.add_force(-force.clone() * one_half);
                    aux2.add_force(force * one_half);
                }

                if let Some(force_result) = c2.calculate_force_between(&p2, &v2, &p1, &v1, &i1) {
                    let force = force_result?;
                    aux1.add_force(force.clone() * one_half);
                    aux2.add_force(-force * one_half);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn calculate_force_between_cells_external<
        Pos,
        Vel,
        For,
        Float,
        Inf,
        F,
        const N: usize,
    >(
        &mut self,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_inf: &Inf,
    ) -> Result<For, CalcError>
    where
        For: Clone
            + core::ops::AddAssign
            + num::Zero
            + core::ops::Mul<F, Output = For>
            + core::ops::Neg<Output = For>,
        C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>
            + cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        A: UpdateMechanics<Pos, Vel, For, F, N>,
        F: num::Float,
    {
        let one_half = F::one() / (F::one() + F::one());
        let mut force = For::zero();
        for (cell, aux_storage) in self.cells.iter_mut() {
            match cell.calculate_force_between(
                &cell.pos(),
                &cell.velocity(),
                &ext_pos,
                &ext_vel,
                &ext_inf,
            ) {
                Some(Ok(f)) => {
                    aux_storage.add_force(-f.clone() * one_half);
                    force += f * one_half;
                }
                Some(Err(e)) => return Err(e),
                None => (),
            };
        }
        Ok(force)
    }

    pub(crate) fn update_cell_cycle_3<Float>(&mut self, dt: &Float) -> Result<(), SimulationError>
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
                    index: index.clone(),
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
    pub(crate) index: I,
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
    pub fn update_cycle<Float>(&mut self, dt: Float) -> Result<(), CalcError>
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
    /// Update cells position and velocity
    ///
    /// We assume that cells implement the
    /// [Mechanics](cellular_raza_concepts::Mechanics) and
    /// [Interaction](cellular_raza_concepts::Interaction) traits.
    /// Then, threads will exchange information in the [PosInformation] format
    /// to calculate the forces acting on the cells.
    pub fn update_mechanics_step_1<Pos, Vel, For, Float, Inf, const N: usize>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        Pos: Clone,
        Vel: Clone,
        Inf: Clone,
        C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>,
        A: UpdateMechanics<Pos, Vel, For, Float, N>,
        For: Clone
            + core::ops::AddAssign
            + core::ops::Mul<Float, Output = For>
            + core::ops::Neg<Output = For>
            + num::Zero,
        Float: num::Float,
        <S as SubDomain<C>>::VoxelIndex: Ord,
        Com: Communicator<SubDomainPlainIndex, PosInformation<Pos, Vel, Inf>>,
    {
        self.voxels
            .iter_mut()
            .map(|(_, vox)| vox.calculate_force_between_cells_internally())
            .collect::<Result<(), CalcError>>()?;
        // Calculate forces for all cells from neighbors
        // TODO can we do this without memory allocation?
        let key_iterator: Vec<_> = self.voxels.keys().map(|k| *k).collect();

        for voxel_index in key_iterator {
            for cell_index_in_vector in 0..self.voxels[&voxel_index].cells.len() {
                let cell_pos = self.voxels[&voxel_index].cells[cell_index_in_vector]
                    .0
                    .pos();
                let cell_vel = self.voxels[&voxel_index].cells[cell_index_in_vector]
                    .0
                    .velocity();
                let cell_inf = self.voxels[&voxel_index].cells[cell_index_in_vector]
                    .0
                    .get_interaction_information();
                let mut force = For::zero();
                let neighbors = self.voxels[&voxel_index].neighbors.clone();
                for neighbor_index in neighbors {
                    match self.voxels.get_mut(&neighbor_index) {
                        Some(vox) => Ok::<(), CalcError>(
                            force += vox.calculate_force_between_cells_external(
                                &cell_pos, &cell_vel, &cell_inf,
                            )?,
                        ),
                        None => Ok(self.communicator.send(
                            &self.plain_index_to_subdomain[&neighbor_index],
                            PosInformation {
                                index_sender: voxel_index,
                                index_receiver: neighbor_index.clone(),
                                pos: cell_pos.clone(),
                                vel: cell_vel.clone(),
                                info: cell_inf.clone(),
                                cell_index_in_vector,
                            },
                        )?),
                    }?;
                }
                self.voxels.get_mut(&voxel_index).unwrap().cells[cell_index_in_vector]
                    .1
                    .add_force(force);
            }
        }

        // Calculate custom force of voxel on cell
        /* TODO
        self.voxels
            .iter_mut()
            .map(|(_, vox)| vox.calculate_custom_force_on_cells())
            .collect::<Result<(), CalcError>>()?;*/
        Ok(())
    }

    /// Update cells position and velocity
    ///
    /// We assume that cells implement the
    /// [Mechanics](cellular_raza_concepts::Mechanics) and
    /// [Interaction](cellular_raza_concepts::Interaction) traits.
    /// Then, threads will use the previously exchanged [PosInformation] to calculate forces
    /// and send back information about the acting force in the [ForceInformation] format.
    /// In addition, this method also applies the inverse force to local cells.
    pub fn update_mechanics_step_2<Pos, Vel, For, Float, Inf, const N: usize>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        For: Clone
            + core::ops::AddAssign
            + num::Zero
            + core::ops::Mul<Float, Output = For>
            + core::ops::Neg<Output = For>,
        C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>
            + cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        A: UpdateMechanics<Pos, Vel, For, Float, N>,
        Float: num::Float,
        Pos: Clone,
        Vel: Clone,
        For: Clone,
        C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>,
        Com: Communicator<SubDomainPlainIndex, PosInformation<Pos, Vel, Inf>>,
        Com: Communicator<SubDomainPlainIndex, ForceInformation<For>>,
    {
        // Receive PositionInformation and send back ForceInformation
        for pos_info in
            <Com as Communicator<SubDomainPlainIndex, PosInformation<Pos, Vel, Inf>>>::receive(
                &mut self.communicator,
            )
            .iter()
        {
            let vox = self.voxels
                .get_mut(&pos_info.index_receiver)
                .ok_or(cellular_raza_concepts::IndexError(format!("EngineError: Voxel with index {:?} of PosInformation can not be found in this thread.", pos_info.index_receiver)))?;
            // Calculate force from cells in voxel
            let force = vox.calculate_force_between_cells_external(
                &pos_info.pos,
                &pos_info.vel,
                &pos_info.info,
            )?;

            // Send back force information
            // let thread_index = self.plain_index_to_subdomain[&pos_info.index_sender];
            self.communicator.send(
                &self.plain_index_to_subdomain[&pos_info.index_sender],
                ForceInformation {
                    force,
                    cell_index_in_vector: pos_info.cell_index_in_vector,
                    index_sender: pos_info.index_sender,
                },
            )?;
        }
        Ok(())
    }

    /// Update cells position and velocity
    ///
    /// We assume that cells implement the
    /// [Mechanics](cellular_raza_concepts::Mechanics) and
    /// [Interaction](cellular_raza_concepts::Interaction) traits.
    /// In this last step, all [ForceInformation] are gathered and used to update the
    /// cells positions and velocities.
    ///
    /// For the future, we hope to provide an abstracted method to use
    /// any of our implemented solvers.
    /// The solver currently limits the number of saved previous increments in the [UpdateMechanics]
    /// trait.
    ///
    /// Currently, we employ the [mechanics_adams_bashforth_3](super::mechanics_adams_bashforth_3)
    /// solver.
    pub fn update_mechanics_step_3<Pos, Vel, For, Inf, Float>(
        &mut self,
        dt: &Float,
    ) -> Result<(), SimulationError>
    where
        A: UpdateMechanics<Pos, Vel, For, Float, 2>,
        Com: Communicator<SubDomainPlainIndex, PosInformation<Pos, Vel, Inf>>,
        Com: Communicator<SubDomainPlainIndex, ForceInformation<For>>,
        C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>
            + cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>
            + Clone,
        Float: num::Float + Copy + FromPrimitive,
        Pos: core::ops::Mul<Float, Output = Pos>,
        Pos: core::ops::Add<Pos, Output = Pos>,
        Pos: core::ops::Sub<Pos, Output = Pos>,
        Pos: Clone,
        Vel: core::ops::Mul<Float, Output = Vel>,
        Vel: core::ops::Add<Vel, Output = Vel>,
        Vel: core::ops::Sub<Vel, Output = Vel>,
        Vel: Clone,
    {
        // Update position and velocity of all cells with new information
        for obt_forces in
            <Com as Communicator<SubDomainPlainIndex, ForceInformation<For>>>::receive(
                &mut self.communicator,
            )
            .into_iter()
        {
            let error_1 = format!(
                "EngineError: Sender with plain index {:?} was ended up in location\
                where index is not present anymore",
                obt_forces.index_sender
            );
            let vox = self
                .voxels
                .get_mut(&obt_forces.index_sender)
                .ok_or(cellular_raza_concepts::IndexError(error_1))?;
            let error_2 = format!(
                "\
                EngineError: Force Information with sender index {:?} and\
                cell at vector position {} could not be matched",
                obt_forces.index_sender, obt_forces.cell_index_in_vector
            );
            match vox.cells.get_mut(obt_forces.cell_index_in_vector) {
                Some((_, aux_storage)) => Ok(aux_storage.add_force(obt_forces.force)),
                None => Err(cellular_raza_concepts::IndexError(error_2)),
            }?;
        }

        self.voxels
            .iter_mut()
            .map(|(_, vox)| {
                vox.cells.iter_mut().map(|(cellbox, aux_storage)| {
                    super::solvers::mechanics_adams_bashforth_3::<C, A, Pos, Vel, For, Float>(
                        cellbox,
                        aux_storage,
                        *dt,
                    )
                })
            })
            .flatten()
            .collect::<Result<Vec<_>, CalcError>>()?;
        Ok(())
    }

    /// Sort new cells into respective voxels
    ///
    /// This step determines if a cell is still in its correct location
    /// after its position has changed. This can be due to the [SubDomainBox::update_mechanics_step_3]
    /// method or due to other effects such as cell-division by the [cellular_raza_concepts::Cycle]
    /// trait.
    ///
    /// If the cell is not in the correct voxel, we either directly insert this cell into the voxel
    /// or send it to the other [SubDomainBox] to take care of this.
    pub fn sort_cells_in_voxels_step_1<Pos, Vel, For, Float>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        Com: Communicator<SubDomainPlainIndex, SendCell<CellBox<C>, A>>,
        S: cellular_raza_concepts::domain_new::SubDomain<C>,
        S::VoxelIndex: Eq + Hash,
    {
        // Store all cells which need to find a new home in this variable
        let mut find_new_home_cells = Vec::<_>::new();

        for (voxel_index, vox) in self.voxels.iter_mut() {
            // Drain every cell which is currently not in the correct voxel
            // TODO use drain_filter when stabilized
            let mut errors = Vec::new();
            let (new_voxel_cells, old_voxel_cells): (Vec<_>, Vec<_>) =
                vox.cells.drain(..).partition(|(c, _)| {
                    let cell_index = self.subdomain.get_voxel_index_of(&c);
                    match cell_index {
                        Ok(index) => {
                            let plain_index = self.voxel_index_to_plain_index[&index];
                            &plain_index != voxel_index
                        }
                        Err(e) => {
                            errors.push(e);
                            false
                        }
                    }
                });
            find_new_home_cells.extend(new_voxel_cells);
            vox.cells = old_voxel_cells;
            /* let new_voxel_cells = vox.cells.drain_filter(|(c, _)| match self.index_to_plain_index.get(&self.domain.get_voxel_index(&c)) {
                Some(ind) => ind,
                None => panic!("Cannot find index {:?}", self.domain.get_voxel_index(&c)),
            }!=voxel_index);
            // Check if the cell needs to be sent to another multivoxelcontainer
            find_new_home_cells.append(&mut new_voxel_cells.collect::<Vec<_>>());*/
        }

        // Send cells to other multivoxelcontainer or keep them here
        for (cell, aux_storage) in find_new_home_cells {
            let ind = self.subdomain.get_voxel_index_of(&cell)?;
            let cell_index = self.voxel_index_to_plain_index[&ind];
            // let new_thread_index = self.index_to_thread[&ind];
            // let cell_index = self.index_to_plain_index[&ind];
            match self.voxels.get_mut(&cell_index) {
                // If new voxel is in current multivoxelcontainer then save them there
                Some(vox) => {
                    vox.cells.push((cell, aux_storage));
                    Ok(())
                }
                // Otherwise send them to the correct other multivoxelcontainer
                None => <Com as Communicator<SubDomainPlainIndex, SendCell<CellBox<C>, A>>>::send(
                    &mut self.communicator,
                    &self.plain_index_to_subdomain[&cell_index],
                    SendCell(cell, aux_storage),
                ),
            }?;
        }
        Ok(())
    }

    /// Sort new cells into respective voxels
    ///
    /// After having sent cells to the new [SubDomainBox] in the
    /// [SubDomainBox::sort_cells_in_voxels_step_1] method, we receive these new cells and insert
    /// them into their respective voxels.
    pub fn sort_cells_in_voxels_step_2(&mut self) -> Result<(), SimulationError>
    where
        Com: Communicator<SubDomainPlainIndex, SendCell<CellBox<C>, A>>,
        S::VoxelIndex: Eq + Hash,
    {
        // Now receive new cells and insert them
        for sent_cell in
            <Com as Communicator<SubDomainPlainIndex, SendCell<CellBox<C>, A>>>::receive(
                &mut self.communicator,
            )
            .into_iter()
        {
            let SendCell(cell, aux_storage) = sent_cell;
            let index =
                self.voxel_index_to_plain_index[&self.subdomain.get_voxel_index_of(&cell)?];

            match self.voxels.get_mut(&index) {
                Some(vox) => Ok(vox.cells.push((cell, aux_storage))),
                None => Err(cellular_raza_concepts::IndexError(format!(
                    "Cell with index {:?} was sent to subdomain which does not hold this index",
                    index
                ))),
            }?;
        }
        Ok(())
    }

    /// Separate function to update the cell cycle
    ///
    /// Instead of running one big update function for all local rules, we have to treat this cell
    /// cycle differently since new cells could be generated and thus have consequences for other
    /// update steps as well.
    pub fn update_cell_cycle_3<Float>(&mut self, dt: &Float) -> Result<(), SimulationError>
    where
        C: cellular_raza_concepts::Cycle<C, Float>
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
    pub fn save_voxels<F>(
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
}
