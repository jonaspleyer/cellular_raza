use cellular_raza_concepts::errors::{BoundaryError, CalcError};
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::hash::Hash;

use rand::SeedableRng;

use super::aux_storage::*;
use super::errors::*;
use super::simulation_flow::*;
use cellular_raza_concepts::domain_new::*;

use super::{CellIdentifier, SubDomainPlainIndex, VoxelPlainIndex};

pub struct SimulationSupervisor<I, Sb> {
    // TODO make this private
    pub subdomain_boxes: HashMap<I, Sb>,
}

/// Simple trait that will be implemented by a
pub trait Supervisor {
    type SimulationResult;

    fn solve_full_system(self) -> Self::SimulationResult;
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
    pub fn calculate_force_between_cells_internally<Pos, Vel, For, Float, Inf, const N: usize>(
        &mut self,
    ) -> Result<(), CalcError>
    where
        C: cellular_raza_concepts::mechanics::Mechanics<Pos, Vel, For, Float>,
        C: cellular_raza_concepts::interaction::Interaction<Pos, Vel, For, Inf>,
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

    pub fn calculate_force_between_cells_external<Pos, Vel, For, Float, Inf, F, const N: usize>(
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
        C: cellular_raza_concepts::interaction::Interaction<Pos, Vel, For, Inf>
            + cellular_raza_concepts::mechanics::Mechanics<Pos, Vel, For, Float>,
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

    pub fn update_cell_cycle_3<Float>(&mut self, dt: &Float) -> Result<(), CalcError>
    where
        C: cellular_raza_concepts::cycle::Cycle<C, Float> + Id,
        A: UpdateCycle + Default,
    {
        use cellular_raza_concepts::cycle::CycleEvent;
        // Update the cell individual cells
        self.cells
            .iter_mut()
            .map(|(cbox, aux_storage)| {
                // Check for cycle events and do update if necessary
                let mut remaining_events = Vec::new();
                for event in aux_storage.get_cycle_events() {
                    match event {
                        CycleEvent::Division => {
                            // TODO catch this error
                            let new_cell = C::divide(&mut self.rng, &mut cbox.cell).unwrap();
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
                    // TODO catch this error!
                    match C::update_conditional_phased_death(&mut self.rng, dt, &mut cbox.cell)
                        .unwrap()
                    {
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
            .collect::<Result<(), CalcError>>()?;

        // Remove cells which are flagged for death
        self.cells.retain(|(_, aux_storage)| {
            !aux_storage.get_cycle_events().contains(&CycleEvent::Remove)
        });

        // Include new cells
        self.cells
            .extend(self.new_cells.drain(..).map(|(cell, parent_id)| {
                self.id_counter += 1;
                (
                    CellBox::new(
                        CellIdentifier(self.plain_index, self.id_counter),
                        parent_id,
                        cell,
                    ),
                    A::default(),
                )
            }));
        Ok(())
    }
}

impl<I, S, C, A, Com, Sy> From<DecomposedDomain<I, S, C>>
    for SimulationSupervisor<I, SubDomainBox<S, C, A, Com, Sy>>
where
    S: SubDomain<C>,
    S::VoxelIndex: Eq + Hash + Ord + Clone,
    I: Eq + PartialEq + core::hash::Hash + Clone + Ord,
    A: Default,
    Sy: super::simulation_flow::FromMap<I>,
    Com: super::simulation_flow::FromMap<I>,
{
    // TODO this is not a BoundaryError
    ///
    fn from(
        decomposed_domain: DecomposedDomain<I, S, C>,
    ) -> SimulationSupervisor<I, SubDomainBox<S, C, A, Com, Sy>> {
        // TODO do not unwrap
        let mut syncers = Sy::from_map(&decomposed_domain.neighbor_map).unwrap();
        let mut communicators = Com::from_map(&decomposed_domain.neighbor_map).unwrap();
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
                    decomposed_domain.voxel_index_to_plain_index[&voxel_index],
                    SubDomainPlainIndex(subdomain_index),
                )
            })
            .collect();

        let subdomain_boxes = decomposed_domain
            .index_subdomain_cells
            .into_iter()
            .map(|(index, subdomain, cells)| {
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
                                .map(|neighbor_index| {
                                    decomposed_domain.voxel_index_to_plain_index[&neighbor_index]
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();
                let voxels = subdomain.get_all_indices().into_iter().map(|voxel_index| {
                    let plain_index = decomposed_domain.voxel_index_to_plain_index[&voxel_index];
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
                let syncer = syncers.remove(&index).ok_or(BoundaryError(
                    "Index was not present in subdomain map".into(),
                ))?;
                let communicator = communicators.remove(&index).ok_or(BoundaryError(
                    "Index was not present in subdomain map".into(),
                ))?;
                let mut subdomain_box = SubDomainBox {
                    plain_index_to_subdomain: plain_index_to_subdomain.clone(),
                    communicator,
                    subdomain,
                    voxels: voxels.collect(),
                    voxel_index_to_plain_index: decomposed_domain
                        .voxel_index_to_plain_index
                        .clone(),
                    syncer,
                };
                subdomain_box.insert_cells(&mut cells)?;
                Ok((index, subdomain_box))
            })
            .collect::<Result<HashMap<_, _>, BoundaryError>>()
            .unwrap();
        let simulation_supervisor = SimulationSupervisor { subdomain_boxes };
        simulation_supervisor
    }
}

/// Encapsulates a subdomain with cells and other simulation aspects.
pub struct SubDomainBox<S, C, A, Com, Sy = BarrierSync>
where
    S: SubDomain<C>,
{
    pub(crate) subdomain: S,
    pub(crate) voxels: std::collections::BTreeMap<VoxelPlainIndex, Voxel<C, A>>,
    pub(crate) voxel_index_to_plain_index:
        std::collections::HashMap<S::VoxelIndex, VoxelPlainIndex>,
    pub(crate) plain_index_to_subdomain:
        std::collections::BTreeMap<VoxelPlainIndex, SubDomainPlainIndex>,
    pub(crate) communicator: Com,
    pub(crate) syncer: Sy,
}

impl<S, C, A, Com, Sy> SubDomainBox<S, C, A, Com, Sy>
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
        for cell in new_cells.drain(..) {
            let voxel_index = self.subdomain.get_voxel_index_of(&cell.0)?;
            let plain_index = self.voxel_index_to_plain_index[&voxel_index];
            let voxel = self.voxels.get_mut(&plain_index).ok_or(BoundaryError(
                "Could not find correct voxel for cell".to_owned(),
            ))?;
            voxel.cells.push((
                CellBox::new(
                    CellIdentifier(voxel.plain_index, voxel.id_counter),
                    None,
                    cell.0,
                ),
                cell.1.map_or(A::default(), |x| x),
            ));
        }
        Ok(())
    }

    /// Advances the cycle of a cell by a small time increment `dt`.
    pub fn update_cycle<Float>(&mut self, dt: Float) -> Result<(), CalcError>
    where
        C: cellular_raza_concepts::cycle::Cycle<C, Float>,
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

impl<S, C, A, Com, Sy> SubDomainBox<S, C, A, Com, Sy>
where
    S: SubDomain<C>,
{
    pub fn update_mechanics_step_1<Pos, Vel, For, Float, Inf, const N: usize>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        Pos: Clone,
        Vel: Clone,
        Inf: Clone,
        C: cellular_raza_concepts::mechanics::Mechanics<Pos, Vel, For, Float>,
        C: cellular_raza_concepts::interaction::Interaction<Pos, Vel, For, Inf>,
        A: UpdateMechanics<Pos, Vel, For, Float, N>,
        For: Clone
            + core::ops::AddAssign
            + core::ops::Mul<Float, Output = For>
            + core::ops::Neg<Output = For>
            + num::Zero,
        Float: num::Float,
        <S as SubDomain<C>>::VoxelIndex: Ord,
        Com: Communicator<VoxelPlainIndex, PosInformation<Pos, Vel, Inf>>,
    {
        self.voxels
            .iter_mut()
            .map(|(_, vox)| vox.calculate_force_between_cells_internally())
            .collect::<Result<(), CalcError>>()?;
        // Calculate forces for all cells from neighbors
        // TODO can we do this without memory allocation?
        let key_iterator: Vec<_> = self.voxels.keys().map(|k| *k).collect();

        for voxel_index in key_iterator {
            for cell_count in 0..self.voxels[&voxel_index].cells.len() {
                let cell_pos = self.voxels[&voxel_index].cells[cell_count].0.pos();
                let cell_vel = self.voxels[&voxel_index].cells[cell_count].0.velocity();
                let cell_inf = self.voxels[&voxel_index].cells[cell_count]
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
                            &neighbor_index,
                            PosInformation {
                                index_sender: voxel_index,
                                index_receiver: neighbor_index.clone(),
                                pos: cell_pos.clone(),
                                vel: cell_vel.clone(),
                                info: cell_inf.clone(),
                                count: cell_count,
                            },
                        )?),
                    }?;
                }
                self.voxels.get_mut(&voxel_index).unwrap().cells[cell_count]
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

    pub fn update_mechanics_step_2<Pos, Vel, For, Float, Inf, const N: usize>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        For: Clone
            + core::ops::AddAssign
            + num::Zero
            + core::ops::Mul<Float, Output = For>
            + core::ops::Neg<Output = For>,
        C: cellular_raza_concepts::interaction::Interaction<Pos, Vel, For, Inf>
            + cellular_raza_concepts::mechanics::Mechanics<Pos, Vel, For, Float>,
        A: UpdateMechanics<Pos, Vel, For, Float, N>,
        Float: num::Float,
        Pos: Clone,
        Vel: Clone,
        For: Clone,
        C: cellular_raza_concepts::mechanics::Mechanics<Pos, Vel, For, Float>,
        C: cellular_raza_concepts::interaction::Interaction<Pos, Vel, For, Inf>,
        Com: Communicator<VoxelPlainIndex, PosInformation<Pos, Vel, Inf>>,
        Com: Communicator<VoxelPlainIndex, ForceInformation<For>>,
    {
        // Receive PositionInformation and send back ForceInformation
        for pos_info in
            <Com as Communicator<VoxelPlainIndex, PosInformation<Pos, Vel, Inf>>>::receive(
                &mut self.communicator,
            )
            .iter()
        {
            let vox = self.voxels
                .get_mut(&pos_info.index_receiver)
                .ok_or(cellular_raza_concepts::errors::IndexError(format!("EngineError: Voxel with index {:?} of PosInformation can not be found in this thread.", pos_info.index_receiver)))?;
            // Calculate force from cells in voxel
            let force = vox.calculate_force_between_cells_external(
                &pos_info.pos,
                &pos_info.vel,
                &pos_info.info,
            )?;

            // Send back force information
            // let thread_index = self.plain_index_to_subdomain[&pos_info.index_sender];
            self.communicator.send(
                &pos_info.index_sender,
                ForceInformation {
                    force,
                    count: pos_info.count,
                    index_sender: pos_info.index_sender,
                },
            )?;
        }
        Ok(())
    }

    pub fn update_mechanics_step_3<Pos, Vel, For, Inf, Float, const N: usize>(
        &mut self,
        dt: &Float,
    ) -> Result<(), SimulationError>
    where
        A: UpdateMechanics<Pos, Vel, For, Float, N>,
        Com: Communicator<VoxelPlainIndex, PosInformation<Pos, Vel, Inf>>,
        Com: Communicator<VoxelPlainIndex, ForceInformation<For>>,
        C: cellular_raza_concepts::interaction::Interaction<Pos, Vel, For, Inf>
            + cellular_raza_concepts::mechanics::Mechanics<Pos, Vel, For, Float>
            + Clone,
        Float: Copy,
    {
        // Update position and velocity of all cells with new information
        for obt_forces in <Com as Communicator<VoxelPlainIndex, ForceInformation<For>>>::receive(
            &mut self.communicator,
        )
        .into_iter()
        {
            let vox = self.voxels.get_mut(&obt_forces.index_sender).ok_or(cellular_raza_concepts::errors::IndexError(format!("EngineError: Sender with plain index {} was ended up in location where index is not present anymore", obt_forces.index_sender)))?;
            match vox.cells.get_mut(obt_forces.count) {
                Some((_, aux_storage)) => Ok(aux_storage.add_force(obt_forces.force)),
                None => Err(cellular_raza_concepts::errors::IndexError(format!("EngineError: Force Information with sender index {:?} and cell at vector position {} could not be matched", obt_forces.index_sender, obt_forces.count))),
            }?;
        }

        self.voxels.iter_mut().for_each(|(_, vox)| {
            vox.cells.iter_mut().for_each(|(cellbox, aux_storage)| {
                super::solvers::mechanics_adams_bashforth::<C, A, Pos, Vel, For, Float, N>(
                    cellbox,
                    aux_storage,
                    *dt,
                )
            });
        });
        /*
        // Update position and velocity of cells
        for (_, vox) in self.voxels.iter_mut() {
            for (cell, aux_storage) in vox.cells.iter_mut() {
                // Calculate the current increment
                let (dx, dv) = cell.calculate_increment(aux_storage.force.clone())?;

                // Use the two-step Adams-Bashforth method. See also: https://en.wikipedia.org/wiki/Linear_multistep_method
                // TODO We should be able to implement arbitrary steppers here
                match (
                    aux_storage.inc_pos_back_1.clone(),
                    aux_storage.inc_pos_back_2.clone(),
                    aux_storage.inc_vel_back_1.clone(),
                    aux_storage.inc_vel_back_2.clone(),
                ) {
                    // If all values are present, use the Adams-Bashforth 3rd order
                    (
                        Some(inc_pos_back_1),
                        Some(inc_pos_back_2),
                        Some(inc_vel_back_1),
                        Some(inc_vel_back_2),
                    ) => {
                        cell.set_pos(
                            &(cell.pos() + dx.clone() * (23.0 / 12.0) * *dt
                                - inc_pos_back_1 * (16.0 / 12.0) * *dt
                                + inc_pos_back_2 * (5.0 / 12.0) * *dt),
                        );
                        cell.set_velocity(
                            &(cell.velocity() + dv.clone() * (23.0 / 12.0) * *dt
                                - inc_vel_back_1 * (16.0 / 12.0) * *dt
                                + inc_vel_back_2 * (5.0 / 12.0) * *dt),
                        );
                    }
                    // Otherwise check and use the 2nd order
                    (Some(inc_pos_back_1), None, Some(inc_vel_back_1), None) => {
                        cell.set_pos(
                            &(cell.pos() + dx.clone() * (3.0 / 2.0) * *dt
                                - inc_pos_back_1 * (1.0 / 2.0) * *dt),
                        );
                        cell.set_velocity(
                            &(cell.velocity() + dv.clone() * (3.0 / 2.0) * *dt
                                - inc_vel_back_1 * (1.0 / 2.0) * *dt),
                        );
                    }
                    // This case should only exists when the cell was first created
                    // Then use the Euler Method
                    _ => {
                        cell.set_pos(&(cell.pos() + dx.clone() * *dt));
                        cell.set_velocity(&(cell.velocity() + dv.clone() * *dt));
                    }
                }

                // Afterwards update values in auxiliary storage
                aux_storage.force = For::zero();
                aux_storage.inc_pos_back_1 = Some(dx);
                aux_storage.inc_vel_back_1 = Some(dv);
            }
        }*/
        Ok(())
    }
}
