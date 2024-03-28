#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{
    CellBox, Communicator, SimulationError, SubDomainBox, SubDomainPlainIndex, UpdateMechanics,
    Voxel, VoxelPlainIndex,
};
use cellular_raza_concepts::{
    domain_new::{SubDomain, SubDomainMechanics},
    BoundaryError, CalcError,
};

/// Send about the position of cells between threads.
///
/// This type is used during the update steps for cellular mechanics
/// [update_mechanics_step_1](super::datastructures::SubDomainBox::update_mechanics_step_1).
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
/// [update_mechanics_step_2](super::datastructures::SubDomainBox::update_mechanics_step_2).
/// The received information is then used in combination with the already present information
/// to update the position and velocity of cells in
/// [update_mechanics_step_3](super::datastructures::SubDomainBox::update_mechanics_step_3).
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
pub struct SendCell<Cel, Aux>(pub Cel, pub Aux);

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

                let force = c1.calculate_force_between(&p1, &v1, &p2, &v2, &i2)?;
                aux1.add_force(-force.clone() * one_half);
                aux2.add_force(force * one_half);

                let force = c2.calculate_force_between(&p2, &v2, &p1, &v1, &i1)?;
                aux1.add_force(force.clone() * one_half);
                aux2.add_force(-force * one_half);
            }
        }
        Ok(())
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
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
            let f = cell.calculate_force_between(
                &cell.pos(),
                &cell.velocity(),
                &ext_pos,
                &ext_vel,
                &ext_inf,
            )?;
            aux_storage.add_force(-f.clone() * one_half);
            force += f * one_half;
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
        <S as SubDomain>::VoxelIndex: Ord,
        S: SubDomainMechanics<Pos, Vel>,
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
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
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
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub fn update_mechanics_step_3<
        Pos,
        Vel,
        For,
        Inf,
        #[cfg(feature = "tracing")] Float: core::fmt::Debug,
        #[cfg(not(feature = "tracing"))] Float,
    >(
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
        Float: num::Float + Copy + num::FromPrimitive,
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

    /// Applies boundary conditions to cells. For the future, we hope to be using previous and current position
    /// of cells rather than the cell itself.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn apply_boundary<Pos, Vel, For, Float>(&mut self) -> Result<(), BoundaryError>
    where
        C: cellular_raza_concepts::Mechanics<Pos, Vel, For, Float>,
        S: SubDomainMechanics<Pos, Vel>,
    {
        self.voxels
            .iter_mut()
            .map(|(_, voxel)| voxel.cells.iter_mut())
            .flatten()
            .map(|(cell, _)| self.subdomain.apply_boundary(cell))
            .collect::<Result<(), BoundaryError>>()
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
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn sort_cells_in_voxels_step_1(&mut self) -> Result<(), SimulationError>
    where
        Com: Communicator<SubDomainPlainIndex, SendCell<CellBox<C>, A>>,
        S: cellular_raza_concepts::domain_new::SubDomainSortCells<C>,
        S::VoxelIndex: Eq + core::hash::Hash,
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
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn sort_cells_in_voxels_step_2(&mut self) -> Result<(), SimulationError>
    where
        Com: Communicator<SubDomainPlainIndex, SendCell<CellBox<C>, A>>,
        S::VoxelIndex: Eq + core::hash::Hash,
        S: cellular_raza_concepts::domain_new::SubDomainSortCells<C>,
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
}
