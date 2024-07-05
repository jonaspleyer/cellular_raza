use super::{
    Communicator, SimulationError, SubDomainBox, SubDomainPlainIndex, UpdateReactions,
    UpdateReactionsContact, Voxel, VoxelPlainIndex,
};
use cellular_raza_concepts::*;

#[cfg(feature = "tracing")]
use tracing::instrument;

impl<I, S, C, A, Com, Sy> SubDomainBox<I, S, C, A, Com, Sy>
where
    S: SubDomain,
{
    ///
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn update_cellular_reactions<ConcGradientExtracellular, ConcTotalExtracellular>(
        &mut self,
        _dt: &f64,
    ) -> Result<(), SimulationError>
// TODO where
    //     ConcTotalExtracellular: std::ops::AddAssign + Mul<f64, Output = ConcVecIntracellular>,
    //     C: CellularReactions<ConcVecIntracellular, ConcVecExtracellular>
    {
        /* self.voxels
        .iter_mut()
        .map(|(_, voxelbox)| {
            voxelbox.cells.iter_mut().map(
                |(cellbox, _aux_storage)| -> Result<(), SimulationError> {
                    let internal_concentration_vector = cellbox.cell.get_intracellular();

                    let external_concentration_vector = voxelbox
                        .voxel
                        .get_extracellular_at_point(&cellbox.cell.pos())?;
                    let (increment_intracellular, increment_extracellular) = cellbox
                        .cell
                        .calculate_intra_and_extracellular_reaction_increment(
                            &internal_concentration_vector,
                            &external_concentration_vector,
                        )?;

                    let cell_volume = cellbox.cell.get_volume();
                    let voxel_volume = voxelbox.voxel.get_volume();
                    let cell_to_voxel_volume = cell_volume / voxel_volume;

                    // aux_storage.intracellular_concentration_increment += increment_intracellular;
                    voxelbox.extracellular_concentration_increments.push((
                        cellbox.cell.pos(),
                        increment_extracellular * cell_to_voxel_volume,
                    ));
                    // TODO these reactions are currently on the same timescale as the fluid-dynamics but we should consider how this may change if we have different time-scales here
                    // ALso the solver is currently simply an euler stepper.
                    // This should be altered to have something like an (adaptive) Runge Kutta or Dopri (or better)
                    cellbox.cell.set_intracellular(
                        internal_concentration_vector + increment_intracellular * *dt,
                    );
                    Ok(())
                },
            )
        })
        .flatten()
        .collect::<Result<(), SimulationError>>()*/
        Ok(())
    }
}

/// TODO
pub struct ReactionsContactInformation<Pos, I, Inf> {
    /// Current position
    pub pos: Pos,
    /// Current intracellular values
    pub intracellular: I,
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

/// TODO
pub struct ReactionsIntracellularInformation<I> {
    /// Increment of intracellular
    pub intracellular: I,
    /// Index of cell in stored vector
    ///
    /// This property works in tandem with [Self::index_sender] in order to send
    /// the calculated information to the correct cell and update its properties.
    pub cell_index_in_vector: usize,
    /// The voxel index where information is returned to
    pub index_sender: VoxelPlainIndex,
}

impl<C, A> Voxel<C, A> {
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub(crate) fn calculate_contact_reactions_between_cells_internally<
        Ri,
        Pos,
        Inf,
        Float,
        const N: usize,
    >(
        &mut self,
    ) -> Result<(), CalcError>
    where
        C: cellular_raza_concepts::ReactionsContact<Ri, Pos, Inf>,
        A: UpdateReactionsContact<Ri, N>,
        Float: num::Float,
    {
        let one_half: Float = Float::one() / (Float::one() + Float::one());

        for n in 0..self.cells.len() {
            for m in n + 1..self.cells.len() {
                /* let mut cells_mut = self.cells.iter_mut();
                let (c1, aux1) = cells_mut.nth(n).unwrap();
                let (c2, aux2) = cells_mut.nth(m - n - 1).unwrap();

                let p1 = c1.pos();
                let v1 = c1.velocity();
                let i1 = c1.get_interaction_information();

                let p2 = c2.pos();
                let v2 = c2.velocity();
                let i2 = c2.get_interaction_information();

                let (force1, force2) = c1.calculate_force_between(&p1, &v1, &p2, &v2, &i2)?;
                aux1.add_force(force1 * one_half);
                aux2.add_force(force2 * one_half);

                let (force2, force1) = c2.calculate_force_between(&p2, &v2, &p1, &v1, &i1)?;
                aux1.add_force(force1 * one_half);
                aux2.add_force(force2 * one_half);

                // Also check for neighbours
                if c1.is_neighbour(&p1, &p2, &i2)? {
                    aux1.incr_current_neighbours(1);
                }
                if c2.is_neighbour(&p2, &p1, &i1)? {
                    aux2.incr_current_neighbours(1);
                }*/
            }
        }
        Ok(())
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub(crate) fn calculate_contact_reactions_between_cells_external<
        Ri,
        Pos,
        Inf,
        Float,
        const N: usize,
    >(
        &mut self,
        ext_pos: &Pos,
        ext_intra: &Ri,
        ext_inf: &Inf,
    ) -> Result<Ri, CalcError>
    where
        C: cellular_raza_concepts::ReactionsContact<Ri, Pos, Inf>,
        A: UpdateReactions<Ri>,
        Float: num::Float,
    {
        /* let one_half = Float::one() / (Float::one() + Float::one());
        let mut force = For::zero();
        for (cell, aux_storage) in self.cells.iter_mut() {
            let (f1, f2) = cell.calculate_force_between(
                &cell.pos(),
                &cell.velocity(),
                &ext_pos,
                &ext_vel,
                &ext_inf,
            )?;
            aux_storage.add_force(f1 * one_half);
            force += f2 * one_half;

            // Check for neighbours
            if cell.is_neighbour(&cell.pos(), &ext_pos, &ext_inf)? {
                aux_storage.incr_current_neighbours(1);
            }
        }
        Ok(force)*/
        todo!()
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
    pub fn update_contact_reactions_step_1<Ri, Pos, Inf, Float, const N: usize>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        Pos: Clone,
        Inf: Clone,
        C: cellular_raza_concepts::ReactionsContact<Ri, Pos, Inf>,
        A: UpdateReactionsContact<Ri, N>,
        Float: num::Float,
        <S as SubDomain>::VoxelIndex: Ord,
        Com: Communicator<SubDomainPlainIndex, ReactionsContactInformation<Ri, Pos, Inf>>,
    {
        /* for (_, vox) in self.voxels.iter_mut() {
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
        }*/

        Ok(())
    }

    /// Calculates the custom [force](SubDomainForce) of
    /// the domain on the cells.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn calculate_reactions_combined<Ri, Pos, Inf, Float, const N: usize>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        Pos: Clone,
        C: cellular_raza_concepts::ReactionsContact<Ri, Pos, Inf>,
        A: UpdateReactions<Ri>,
        // TODO use the correct trait here (which is not yet developed)
        // S: SubDomainForce<Pos, Vel, For>,
    {
        /* for (cell, aux) in self
            .voxels
            .iter_mut()
            .map(|(_, vox)| vox.cells.iter_mut())
            .flatten()
        {
            let f = self
                .subdomain
                .calculate_custom_force(&cell.pos(), &cell.velocity())?;
            aux.add_force(f);
        }*/
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
    pub fn update_contact_reactions_step_2<Ri, Pos, Inf, Float, const N: usize>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        C: cellular_raza_concepts::ReactionsContact<Ri, Pos, Inf>,
        A: UpdateReactions<Ri>,
        Float: num::Float,
        Pos: Clone,
        Com: Communicator<SubDomainPlainIndex, ReactionsContactInformation<Ri, Pos, Inf>>,
        Com: Communicator<SubDomainPlainIndex, ReactionsIntracellularInformation<Ri>>,
    {
        // Receive PositionInformation and send back ForceInformation
        /* for pos_info in
            <Com as Communicator<SubDomainPlainIndex, PosInformation<Pos, Vel, Inf>>>::receive(
                &mut self.communicator,
            )
            .iter()
        {
            let vox = self.voxels.get_mut(&pos_info.index_receiver).ok_or(
                cellular_raza_concepts::IndexError(format!(
                    "EngineError: Voxel with index {:?} of PosInformation can not be\
                            found in this thread.",
                    pos_info.index_receiver
                )),
            )?;
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
        }*/
        Ok(())
    }

    /// Receive all calculated forces and include them for later update steps.
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub fn update_reactions_contact_step_3<Ri, Pos, Inf>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        A: UpdateReactions<Ri>,
        Com: Communicator<SubDomainPlainIndex, ReactionsIntracellularInformation<Ri>>,
    {
        // Update position and velocity of all cells with new information
        for obt_intracellular in <Com as Communicator<
            SubDomainPlainIndex,
            ReactionsIntracellularInformation<Ri>,
        >>::receive(&mut self.communicator)
        .into_iter()
        {
            let error_1 = format!(
                "EngineError: Sender with plain index {:?} was ended up in location\
                where index is not present anymore",
                obt_intracellular.index_sender
            );
            let vox = self
                .voxels
                .get_mut(&obt_intracellular.index_sender)
                .ok_or(cellular_raza_concepts::IndexError(error_1))?;
            let error_2 = format!(
                "\
                EngineError: Force Information with sender index {:?} and\
                cell at vector position {} could not be matched",
                obt_intracellular.index_sender, obt_intracellular.cell_index_in_vector
            );
            match vox.cells.get_mut(obt_intracellular.cell_index_in_vector) {
                Some((_, aux_storage)) => Ok(aux_storage.incr_conc(obt_intracellular.intracellular)),
                None => Err(cellular_raza_concepts::IndexError(error_2)),
            }?;
        }
        Ok(())
    }
}

/// TODO
pub fn local_reactions_update_step_3<
    C,
    A,
    Ri,
    #[cfg(feature = "tracing")] Float: core::fmt::Debug,
    #[cfg(not(feature = "tracing"))] Float,
>(
    cell: &mut C,
    aux_storage: &mut A,
    dt: Float,
    _rng: &mut rand_chacha::ChaCha8Rng,
) -> Result<(), SimulationError>
where
    A: UpdateReactions<Ri>,
    C: cellular_raza_concepts::Reactions<Ri>,
    Float: num::Float + Copy + num::FromPrimitive,
{
    /* super::solvers::mechanics_adams_bashforth_3::<C, A, Pos, Vel, For, Float>(
        cell,
        aux_storage,
        dt,
    )?;*/
    Ok(())
}
