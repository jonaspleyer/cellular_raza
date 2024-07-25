use super::{
    reactions_contact_adams_bashforth_3rd, reactions_intracellular_runge_kutta_4th, Communicator,
    SimulationError, SubDomainBox, SubDomainPlainIndex, UpdateReactions, UpdateReactionsContact,
    Voxel, VoxelPlainIndex,
};
use cellular_raza_concepts::*;

use num::FromPrimitive;
#[cfg(feature = "tracing")]
use tracing::instrument;

/* impl<I, S, C, A, Com, Sy> SubDomainBox<I, S, C, A, Com, Sy>
where
    S: SubDomain,
{
    ///
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn update_contact_reactions<Ri, Pos, Inf>(
        &mut self,
        _dt: &f64,
    ) -> Result<(), SimulationError>
    where
        C: ReactionsContact<Ri, Pos, Inf>,
    {
        self.voxels
            .iter_mut()
            .map(|(_, voxelbox)| {
                voxelbox.cells.iter_mut().map(
                    |(cell, aux_storage)| -> Result<(), SimulationError> {
                        let intracellular = cell.get_intracellular();
                        Ok(())
                    },
                )
            })
            .flatten()
            .collect::<Result<(), SimulationError>>()?;
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
}*/

/// TODO
pub struct ReactionsContactInformation<Pos, Ri, RInf> {
    /// Current position
    pub pos: Pos,
    /// Current intracellular values
    pub intracellular: Ri,
    /// Information shared between cells
    pub info: RInf,
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
pub struct ReactionsContactReturn<Ri> {
    /// Increment of intracellular
    pub intracellular: Ri,
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
        RInf,
        Float,
        const N: usize,
    >(
        &mut self,
    ) -> Result<(), CalcError>
    where
        C: cellular_raza_concepts::ReactionsContact<Ri, Pos, Float, RInf>,
        C: Position<Pos>,
        Ri: cellular_raza_concepts::Xapy<Float>,
        Ri: num::Zero,
        A: UpdateReactionsContact<Ri, N>,
        A: UpdateReactions<Ri>,
        Float: num::Float,
    {
        let one_half: Float = Float::one() / (Float::one() + Float::one());

        for n in 0..self.cells.len() {
            for m in n + 1..self.cells.len() {
                let mut cells_mut = self.cells.iter_mut();
                let (c1, aux1) = cells_mut.nth(n).unwrap();
                let (c2, aux2) = cells_mut.nth(m - n - 1).unwrap();

                let p1 = c1.pos();
                let intra1 = c1.get_intracellular();
                let rinf1 = c1.get_contact_information();
                let p2 = c2.pos();
                let intra2 = c2.get_intracellular();
                let rinf2 = c2.get_contact_information();

                let (dintra11, dintra12) =
                    c1.calculate_contact_increment(&intra1, &intra2, &p1, &p2, &rinf2)?;
                let (dintra22, dintra21) =
                    c2.calculate_contact_increment(&intra2, &intra1, &p2, &p1, &rinf1)?;

                aux1.incr_conc(dintra11.xapy(one_half, &dintra21.xapy(one_half, &Ri::zero())));
                aux2.incr_conc(dintra22.xapy(one_half, &dintra12.xapy(one_half, &Ri::zero())));
            }
        }
        Ok(())
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub(crate) fn calculate_contact_reactions_between_cells_external<
        Ri,
        Pos,
        RInf,
        Float,
        const N: usize,
    >(
        &mut self,
        ext_pos: &Pos,
        ext_intra: &Ri,
        ext_rinf: &RInf,
    ) -> Result<Ri, CalcError>
    where
        C: cellular_raza_concepts::ReactionsContact<Ri, Pos, Float, RInf>,
        C: Position<Pos>,
        Ri: num::Zero,
        Ri: Xapy<Float>,
        A: UpdateReactions<Ri>,
        A: UpdateReactionsContact<Ri, N>,
        Float: num::Float,
    {
        let one_half = Float::one() / (Float::one() + Float::one());
        let mut dextra_total = Ri::zero();
        for (cell, aux_storage) in self.cells.iter_mut() {
            let own_intra = cell.get_intracellular();
            let own_pos = cell.pos();

            let (dintra, dextra) = cell.calculate_contact_increment(
                &own_intra, &ext_intra, &own_pos, &ext_pos, &ext_rinf,
            )?;
            aux_storage.incr_conc(dintra.xapy(one_half, &Ri::zero()));
            dextra_total = dextra.xapy(one_half, &dextra_total);
        }
        Ok(dextra_total)
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
    pub fn update_contact_reactions_step_1<Ri, Pos, RInf, Float, const N: usize>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        Pos: Clone,
        C: cellular_raza_concepts::ReactionsContact<Ri, Pos, Float, RInf>,
        C: cellular_raza_concepts::Position<Pos>,
        A: UpdateReactions<Ri>,
        A: UpdateReactionsContact<Ri, N>,
        Ri: num::Zero + Xapy<Float> + Clone,
        RInf: Clone,
        Float: num::Float,
        // <S as SubDomain>::VoxelIndex: Ord,
        Com: Communicator<SubDomainPlainIndex, ReactionsContactInformation<Pos, Ri, RInf>>,
    {
        for (_, vox) in self.voxels.iter_mut() {
            vox.calculate_contact_reactions_between_cells_internally::<Ri, Pos, RInf, Float, N>()?;
        }

        // TODO can we do this without memory allocation?
        // or simply allocate when creating the subdomain
        let key_iterator: Vec<_> = self.voxels.keys().map(|k| *k).collect();

        for voxel_index in key_iterator {
            for cell_index_in_vector in 0..self.voxels[&voxel_index].cells.len() {
                let cell_pos = self.voxels[&voxel_index].cells[cell_index_in_vector]
                    .0
                    .pos();
                let cell_intra = self.voxels[&voxel_index].cells[cell_index_in_vector]
                    .0
                    .get_intracellular();
                let cell_contact_inf = self.voxels[&voxel_index].cells[cell_index_in_vector]
                    .0
                    .get_contact_information();
                let mut incr = Ri::zero();
                // TODO can we do this without cloning at all?
                let neighbors = self.voxels[&voxel_index].neighbors.clone();
                for neighbor_index in neighbors {
                    match self.voxels.get_mut(&neighbor_index) {
                        Some(vox) => Ok::<(), CalcError>(
                            incr = incr.xapy(
                                Float::one(),
                                &vox.calculate_contact_reactions_between_cells_external(
                                    &cell_pos,
                                    &cell_intra,
                                    &cell_contact_inf,
                                )?,
                            ),
                            // Ok(())
                        ),
                        None => Ok(self.communicator.send(
                            &self.plain_index_to_subdomain[&neighbor_index],
                            ReactionsContactInformation {
                                pos: cell_pos.clone(),
                                intracellular: cell_intra.clone(),
                                info: cell_contact_inf.clone(),
                                cell_index_in_vector,
                                index_sender: voxel_index,
                                index_receiver: neighbor_index.clone(),
                            },
                        )?),
                    }?;
                }
                self.voxels.get_mut(&voxel_index).unwrap().cells[cell_index_in_vector]
                    .1
                    .incr_conc(incr);
            }
        }
        Ok(())
    }

    /// TODO
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn update_contact_reactions_step_2<Ri, Pos, RInf, Float, const N: usize>(
        &mut self,
        determinism: bool,
    ) -> Result<(), SimulationError>
    where
        C: cellular_raza_concepts::ReactionsContact<Ri, Pos, Float, RInf> + Position<Pos>,
        A: UpdateReactions<Ri> + UpdateReactionsContact<Ri, N>,
        Ri: Xapy<Float> + num::Zero,
        Float: num::Float,
        Pos: Clone,
        Com: Communicator<SubDomainPlainIndex, ReactionsContactInformation<Pos, Ri, RInf>>,
        Com: Communicator<SubDomainPlainIndex, ReactionsContactReturn<Ri>>,
    {
        // Receive contactinformation and send back increments
        let mut received_infos = <Com as Communicator<
            SubDomainPlainIndex,
            ReactionsContactInformation<Pos, Ri, RInf>,
        >>::receive(&mut self.communicator);
        if determinism {
            received_infos.sort_by_key(|info| info.index_sender);
        }
        for contact_info in received_infos {
            let vox = self.voxels.get_mut(&contact_info.index_receiver).ok_or(
                cellular_raza_concepts::IndexError(format!(
                    "EngineError: Voxel with index {:?} of ReactionsContactInformation can not be\
                    found in this threads.",
                    contact_info.index_receiver
                )),
            )?;
            // Calculate the contact increments from cells in voxel
            let incr = vox.calculate_contact_reactions_between_cells_external(
                &contact_info.pos,
                &contact_info.intracellular,
                &contact_info.info,
            )?;

            // Send back information about the increment
            self.communicator.send(
                &self.plain_index_to_subdomain[&contact_info.index_sender],
                ReactionsContactReturn {
                    intracellular: incr,
                    cell_index_in_vector: contact_info.cell_index_in_vector,
                    index_sender: contact_info.index_sender,
                },
            )?;
        }
        Ok(())
    }

    /// Receive all calculated increments and include them for later update steps.
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub fn update_contact_reactions_step_3<Ri>(
        &mut self,
        determinism: bool,
    ) -> Result<(), SimulationError>
    where
        A: UpdateReactions<Ri>,
        Com: Communicator<SubDomainPlainIndex, ReactionsContactReturn<Ri>>,
    {
        // Update position and velocity of all cells with new information
        let mut received_infos = <Com as Communicator<
            SubDomainPlainIndex,
            ReactionsContactReturn<Ri>,
        >>::receive(&mut self.communicator);
        if determinism {
            received_infos.sort_by_key(|info| info.index_sender);
        }
        for obt_intracellular in received_infos {
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
                Some((_, aux_storage)) => {
                    Ok(aux_storage.incr_conc(obt_intracellular.intracellular))
                }
                None => Err(cellular_raza_concepts::IndexError(error_2)),
            }?;
        }
        Ok(())
    }
}

/// Updates the cells intracellular values from the obtained contact informations
pub fn local_update_contact_reactions_step_3<
    C,
    A,
    Ri,
    #[cfg(feature = "tracing")] F: core::fmt::Debug,
    #[cfg(not(feature = "tracing"))] F,
>(
    cell: &mut C,
    aux_storage: &mut A,
    dt: F,
    _rng: &mut rand_chacha::ChaCha8Rng,
) -> Result<(), SimulationError>
where
    A: UpdateReactions<Ri> + UpdateReactionsContact<Ri, 3>,
    C: cellular_raza_concepts::Intracellular<Ri>,
    F: num::Float + FromPrimitive,
    Ri: num::Zero + Xapy<F> + Clone,
{
    reactions_contact_adams_bashforth_3rd(cell, aux_storage, dt)?;
    Ok(())
}

/// TODO
pub fn local_reactions_intracellular<
    C,
    A,
    Ri,
    #[cfg(feature = "tracing")] F: core::fmt::Debug,
    #[cfg(not(feature = "tracing"))] F,
>(
    cell: &mut C,
    aux_storage: &mut A,
    dt: F,
    _rng: &mut rand_chacha::ChaCha8Rng,
) -> Result<(), SimulationError>
where
    A: UpdateReactions<Ri>,
    C: cellular_raza_concepts::Reactions<Ri>,
    F: num::Float,
    Ri: num::Zero + Xapy<F>,
{
    reactions_intracellular_runge_kutta_4th(cell, aux_storage, dt)?;
    Ok(())
}

/// TODO
pub fn local_reactions_clear_increment<
    C,
    A,
    Ri,
    #[cfg(feature = "tracing")] F: core::fmt::Debug,
    #[cfg(not(feature = "tracing"))] F,
>(
    _cell: &mut C,
    aux_storage: &mut A,
    _dt: F,
    _rng: &mut rand_chacha::ChaCha8Rng,
) -> Result<(), SimulationError>
where
    A: UpdateReactions<Ri>,
    Ri: num::Zero,
{
    aux_storage.set_conc(Ri::zero());
    Ok(())
}
