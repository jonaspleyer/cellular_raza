#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{
    AdamsBashforth, CellBox, Communicator, MechanicsAdamsBashforthSolver, SimulationError,
    SubDomainBox, SubDomainPlainIndex, UpdateInteraction, UpdateMechanics, Voxel, VoxelPlainIndex,
};
use cellular_raza_concepts::*;

/// Send about the position of cells between threads.
///
/// This type is used during the update steps for cellular mechanics
/// [update_mechanics_interaction_step_1](super::datastructures::SubDomainBox::update_mechanics_interaction_step_1).
/// The response to [PosInformation] is the [ForceInformation] type.
/// Upon requesting the acting force, by providing the information stored in this struct,
/// the requester obtains the needed information about acting forces.
/// See also the [cellular_raza_concepts::Interaction] trait.
pub struct PosInformation<Pos, Vel, Inf> {
    /// Current position
    pub pos: Pos,
    /// Current velocity
    pub vel: Vel,
    /// Information shared between cells
    pub info: Inf,
    /// Index of cell in stored vector
    ///
    /// When returning information, this property is needed in order
    /// to get the correct cell in the vector of cells and update its properties.
    pub cell_index_in_vector: usize,
    /// Voxel index of the sending cell.
    /// Information should be returned to this voxel.
    pub index_sender: VoxelPlainIndex,
    /// Voxel index of the voxel from which information is requested.
    /// This index is irrelevant after the initial query has been sent.
    pub index_receiver: VoxelPlainIndex,
}

/// Return type to the requested [PosInformation].
///
/// This type is returned after performing all necessary force calculations in
/// [update_mechanics_interaction_step_2](super::datastructures::SubDomainBox::update_mechanics_interaction_step_2).
/// The received information is then used in combination with the already present information
/// to update the position and velocity of cells in
/// [update_mechanics_interaction_step_3](super::datastructures::SubDomainBox::update_mechanics_interaction_step_3).
pub struct ForceInformation<For> {
    /// Overall force acting on cell.
    ///
    /// This force is already combined in the sense that multiple forces may be added together.
    pub force: For,
    /// Index of cell in stored vector
    ///
    /// This property works in tandem with [Self::index_sender] in order to send
    /// the calculated information to the correct cell and update its properties.
    pub cell_index_in_vector: usize,
    /// The voxel index where information is returned to
    pub index_sender: VoxelPlainIndex,
}

/// Send cell and its AuxStorage between threads.
pub struct SendCell<Cel, Aux>(pub VoxelPlainIndex, pub Cel, pub Aux);

impl<C, A> Voxel<C, A> {
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
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
        C: cellular_raza_concepts::Position<Pos>,
        C: cellular_raza_concepts::Velocity<Vel>,
        C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>,
        A: UpdateMechanics<Pos, Vel, For, N>,
        A: UpdateInteraction,
        For: Xapy<Float>,
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

                let (force1, force2) = c1.calculate_force_between(&p1, &v1, &p2, &v2, &i2)?;
                aux1.add_force(force1.xa(one_half));
                aux2.add_force(force2.xa(one_half));

                let (force2, force1) = c2.calculate_force_between(&p2, &v2, &p1, &v1, &i1)?;
                aux1.add_force(force1.xa(one_half));
                aux2.add_force(force2.xa(one_half));

                // Also check for neighbors
                if c1.is_neighbor(&p1, &p2, &i2)? {
                    aux1.incr_current_neighbors(1);
                }
                if c2.is_neighbor(&p2, &p1, &i1)? {
                    aux2.incr_current_neighbors(1);
                }
            }
        }
        Ok(())
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub(crate) fn calculate_force_between_cells_external<
        Pos,
        Vel,
        For,
        Inf,
        Float,
        const N: usize,
    >(
        &mut self,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_inf: &Inf,
    ) -> Result<Option<For>, CalcError>
    where
        C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>
            + cellular_raza_concepts::Position<Pos>
            + cellular_raza_concepts::Velocity<Vel>,
        A: UpdateMechanics<Pos, Vel, For, N>,
        A: UpdateInteraction,
        Float: num::Float,
        For: Xapy<Float> + core::ops::AddAssign,
    {
        use core::borrow::BorrowMut;
        let one_half = Float::one() / (Float::one() + Float::one());
        let mut force = None;
        for (cell, aux_storage) in self.cells.iter_mut() {
            let (f1, f2) = cell.calculate_force_between(
                &cell.pos(),
                &cell.velocity(),
                &ext_pos,
                &ext_vel,
                &ext_inf,
            )?;
            aux_storage.add_force(f1.xa(one_half));
            if let Some(f) = force.borrow_mut() {
                *f = f2.xapy(one_half, &*f);
            } else {
                force = Some(f2.xa(one_half));
            }

            // Check for neighbors
            if cell.is_neighbor(&cell.pos(), &ext_pos, &ext_inf)? {
                aux_storage.incr_current_neighbors(1);
            }
        }
        Ok(force)
    }
}

impl<I, S, C, A, Com, Sy> SubDomainBox<I, S, C, A, Com, Sy>
where
    S: SubDomain,
{
    /// Update cells position and velocity
    ///
    /// We assume that cells implement the
    /// [Mechanics](cellular_raza_concepts::Mechanics) and
    /// [Interaction](cellular_raza_concepts::Interaction) traits.
    /// Then, threads will exchange information in the [PosInformation] format
    /// to calculate the forces acting on the cells.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn update_mechanics_interaction_step_1<Pos, Vel, For, Float, Inf, const N: usize>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        Pos: Clone,
        Vel: Clone,
        Inf: Clone,
        C: cellular_raza_concepts::Position<Pos>,
        C: cellular_raza_concepts::Velocity<Vel>,
        C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>,
        A: UpdateMechanics<Pos, Vel, For, N>,
        A: UpdateInteraction,
        For: Xapy<Float> + core::ops::AddAssign,
        Float: num::Float + core::ops::AddAssign,
        <S as SubDomain>::VoxelIndex: Ord,
        S: SubDomainMechanics<Pos, Vel>,
        Com: Communicator<SubDomainPlainIndex, PosInformation<Pos, Vel, Inf>>,
    {
        for (_, vox) in self.voxels.iter_mut() {
            vox.calculate_force_between_cells_internally()?;
        }

        // Calculate forces for all cells from neighbors
        // TODO can we do this without memory allocation?
        // or simply allocate when creating the subdomain
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
                let mut force = None;
                let neighbors = self.voxels[&voxel_index].neighbors.clone();
                for neighbor_index in neighbors {
                    match self.voxels.get_mut(&neighbor_index) {
                        Some(vox) => {
                            if let Some(f) = vox.calculate_force_between_cells_external(
                                &cell_pos, &cell_vel, &cell_inf,
                            )? {
                                match &mut force {
                                    Some(f2) => *f2 = f.xapy(Float::one(), &f2),
                                    f2 @ None => *f2 = Some(f),
                                }
                            }
                            Ok::<(), CalcError>(())
                        }
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
                if let Some(f) = force {
                    self.voxels.get_mut(&voxel_index).unwrap().cells[cell_index_in_vector]
                        .1
                        .add_force(f);
                }
            }
        }

        Ok(())
    }

    /// Calculates the custom [force](SubDomainForce) of
    /// the domain on the cells.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn calculate_custom_domain_force<Pos, Vel, For, const N: usize>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        Pos: Clone,
        Vel: Clone,
        C: cellular_raza_concepts::Position<Pos>,
        C: cellular_raza_concepts::Velocity<Vel>,
        A: UpdateMechanics<Pos, Vel, For, N>,
        S: SubDomainForce<Pos, Vel, For>,
    {
        for (cell, aux) in self
            .voxels
            .iter_mut()
            .map(|(_, vox)| vox.cells.iter_mut())
            .flatten()
        {
            let f = self
                .subdomain
                .calculate_custom_force(&cell.pos(), &cell.velocity())?;
            aux.add_force(f);
        }
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
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn update_mechanics_interaction_step_2<Pos, Vel, For, Float, Inf, const N: usize>(
        &mut self,
        determinism: bool,
    ) -> Result<(), SimulationError>
    where
        For: Xapy<Float>,
        A: UpdateMechanics<Pos, Vel, For, N>,
        A: UpdateInteraction,
        Float: num::Float,
        Pos: Clone,
        Vel: Clone,
        For: Clone + core::ops::AddAssign,
        C: cellular_raza_concepts::Position<Pos>,
        C: cellular_raza_concepts::Velocity<Vel>,
        C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>,
        Com: Communicator<SubDomainPlainIndex, PosInformation<Pos, Vel, Inf>>,
        Com: Communicator<SubDomainPlainIndex, ForceInformation<For>>,
    {
        // Receive PositionInformation and send back ForceInformation
        let mut received_infos = <Com as Communicator<
            SubDomainPlainIndex,
            PosInformation<Pos, Vel, Inf>,
        >>::receive(&mut self.communicator);
        if determinism {
            received_infos.sort_by_key(|pos_info| pos_info.index_sender);
        }
        for pos_info in received_infos.iter() {
            let vox = self.voxels.get_mut(&pos_info.index_receiver).ok_or(
                cellular_raza_concepts::IndexError(format!(
                    "EngineError: Voxel with index {:?} of PosInformation can not be\
                            found in this thread.",
                    pos_info.index_receiver
                )),
            )?;
            // Calculate force from cells in voxel
            if let Some(force) = vox.calculate_force_between_cells_external(
                &pos_info.pos,
                &pos_info.vel,
                &pos_info.info,
            )? {
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
        }
        Ok(())
    }

    /// Receive all calculated forces and include them for later update steps.
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub fn update_mechanics_interaction_step_3<Pos, Vel, For, const N: usize>(
        &mut self,
        determinism: bool,
    ) -> Result<(), SimulationError>
    where
        A: UpdateMechanics<Pos, Vel, For, N>,
        Com: Communicator<SubDomainPlainIndex, ForceInformation<For>>,
    {
        // Update position and velocity of all cells with new information
        let mut received_infos = <Com as Communicator<
            SubDomainPlainIndex,
            ForceInformation<For>,
        >>::receive(&mut self.communicator);
        if determinism {
            received_infos.sort_by_key(|force_info| force_info.index_sender);
        }
        for obt_forces in received_infos {
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
                EngineError: Force Information with sender index {:?} and \
                cell at vector position {} could not be matched",
                obt_forces.index_sender, obt_forces.cell_index_in_vector
            );
            match vox.cells.get_mut(obt_forces.cell_index_in_vector) {
                Some((_, aux_storage)) => Ok(aux_storage.add_force(obt_forces.force)),
                None => Err(cellular_raza_concepts::IndexError(error_2)),
            }?;
        }
        Ok(())
    }

    /// Applies boundary conditions to cells. For the future, we hope to be using previous and
    /// current position of cells rather than the cell itself.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn apply_boundary<Pos, Vel>(&mut self) -> Result<(), BoundaryError>
    where
        C: cellular_raza_concepts::Position<Pos>,
        C: cellular_raza_concepts::Velocity<Vel>,
        S: SubDomainMechanics<Pos, Vel>,
    {
        for (cell, _) in self
            .voxels
            .iter_mut()
            .map(|(_, voxel)| voxel.cells.iter_mut())
            .flatten()
        {
            let mut pos = cell.pos();
            let mut vel = cell.velocity();
            self.subdomain.apply_boundary(&mut pos, &mut vel)?;
            cell.set_pos(&pos);
            cell.set_velocity(&vel);
        }
        Ok(())
    }

    /// Sort new cells into respective voxels
    ///
    /// This step determines if a cell is still in its correct location
    /// after its position has changed. This can be due to the
    /// [SubDomainBox::update_mechanics_interaction_step_3] method or due to other effects such as
    /// cell-division by the [cellular_raza_concepts::Cycle] trait.
    ///
    /// If the cell is not in the correct voxel, we either directly insert this cell into the voxel
    /// or send it to the other [SubDomainBox] to take care of this.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn sort_cells_in_voxels_step_1(&mut self) -> Result<(), SimulationError>
    where
        Com: Communicator<SubDomainPlainIndex, SendCell<CellBox<C>, A>>,
        S: SortCells<C, VoxelIndex = <S as SubDomain>::VoxelIndex>,
        <S as SubDomain>::VoxelIndex: Eq + core::hash::Hash + Ord,
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
            find_new_home_cells.extend(new_voxel_cells.into_iter().map(|c| (*voxel_index, c)));
            vox.cells = old_voxel_cells;
        }

        // Send cells to other multivoxelcontainer or keep them here
        for (voxel_index, (cell, aux_storage)) in find_new_home_cells {
            let ind = self.subdomain.get_voxel_index_of(&cell)?;
            let cell_index = self.voxel_index_to_plain_index[&ind];
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
                    SendCell(voxel_index, cell, aux_storage),
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
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn sort_cells_in_voxels_step_2(&mut self, determinism: bool) -> Result<(), SimulationError>
    where
        Com: Communicator<SubDomainPlainIndex, SendCell<CellBox<C>, A>>,
        <S as SubDomain>::VoxelIndex: Eq + core::hash::Hash + Ord,
        S: SortCells<C, VoxelIndex = <S as SubDomain>::VoxelIndex>,
    {
        // Now receive new cells and insert them
        let mut received_cells = <Com as Communicator<
            SubDomainPlainIndex,
            SendCell<CellBox<C>, A>,
        >>::receive(&mut self.communicator);
        if determinism {
            received_cells.sort_by_key(|send_cell| send_cell.0);
        }
        for sent_cell in received_cells {
            let SendCell(_, cell, aux_storage) = sent_cell;
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
}

/// We assume that cells implement the
/// [Mechanics](cellular_raza_concepts::Mechanics) and
/// [Interaction](cellular_raza_concepts::Interaction) traits.
/// In this last step, all [ForceInformation] are gathered and used to update the
/// cells positions and velocities.
///
/// For the future, we hope to provide an abstracted method to use any of our implemented
/// solvers.
/// The solver currently limits the number of saved previous increments in the
/// [UpdateMechanics] trait.
///
/// Currently, we employ the [mechanics_adams_bashforth_3](super::mechanics_adams_bashforth_3)
/// solver.
#[allow(private_bounds)]
pub fn local_mechanics_update<
    C,
    A,
    Pos,
    Vel,
    For,
    #[cfg(feature = "tracing")] Float: core::fmt::Debug,
    #[cfg(not(feature = "tracing"))] Float,
    const N: usize,
>(
    cell: &mut C,
    aux_storage: &mut A,
    dt: Float,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> Result<(), SimulationError>
where
    A: UpdateMechanics<Pos, Vel, For, N>,
    C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float> + Clone,
    C: cellular_raza_concepts::Position<Pos>,
    C: cellular_raza_concepts::Velocity<Vel>,
    Float: num::Float + Copy + num::FromPrimitive,
    Pos: Xapy<Float> + Clone,
    Vel: Xapy<Float> + Clone,
    Vel: Clone,
    MechanicsAdamsBashforthSolver<N>: AdamsBashforth<N>,
{
    use super::solvers::{AdamsBashforth, MechanicsAdamsBashforthSolver};
    <MechanicsAdamsBashforthSolver<N> as AdamsBashforth<N>>::update(cell, aux_storage, dt, rng)?;
    Ok(())
}

/// Perform the [Interaction::react_to_neighbors] function and clear current neighbors.
pub fn local_interaction_react_to_neighbors<C, A, Pos, Vel, For, Inf, Float>(
    cell: &mut C,
    aux_storage: &mut A,
    _dt: Float,
    _rng: &mut rand_chacha::ChaCha8Rng,
) -> Result<(), cellular_raza_concepts::CalcError>
where
    C: cellular_raza_concepts::Interaction<Pos, Vel, For, Inf>,
    C: cellular_raza_concepts::Position<Pos>,
    A: UpdateInteraction,
{
    cell.react_to_neighbors(aux_storage.get_current_neighbors())?;
    aux_storage.set_current_neighbors(0);
    Ok(())
}
