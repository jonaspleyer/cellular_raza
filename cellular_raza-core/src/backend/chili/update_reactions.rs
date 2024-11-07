use super::{
    reactions_contact_adams_bashforth_3rd, Communicator, ReactionsRungeKuttaSolver, RungeKutta,
    SimulationError, SubDomainBox, SubDomainPlainIndex, UpdateReactions, UpdateReactionsContact,
    Voxel, VoxelPlainIndex,
};
use cellular_raza_concepts::*;

use num::FromPrimitive;
#[cfg(feature = "tracing")]
use tracing::instrument;

/// Carries information about the border given by the [ReactionsExtra] trait between subdomains.
pub struct ReactionsExtraBorderInfo<Binfo>(pub SubDomainPlainIndex, pub Binfo);

/// Return information of border value after having obtained the [ReactionsExtra::BorderInfo]
pub struct ReactionsExtraBorderReturn<Bvalue>(pub SubDomainPlainIndex, pub Bvalue);

impl<I, S, C, A, Com, Sy> SubDomainBox<I, S, C, A, Com, Sy>
where
    S: SubDomain,
{
    /// TODO
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn update_reactions_extra_step_1<Pos, Ri, Re, Float>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        C: ReactionsExtra<Ri, Re>,
        S: SubDomainReactions<Pos, Re, Float>,
        Com: Communicator<
            SubDomainPlainIndex,
            ReactionsExtraBorderInfo<<S as SubDomainReactions<Pos, Re, Float>>::BorderInfo>,
        >,
    {
        for neighbor_index in self.neighbors.iter() {
            let border_info = self.subdomain.get_border_info();
            self.communicator.send(
                neighbor_index,
                ReactionsExtraBorderInfo(self.subdomain_plain_index, border_info),
            )?;
        }
        Ok(())
    }

    /// TODO
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn update_reactions_extra_step_2<Pos, Ri, Re, Float>(
        &mut self,
        determinism: bool,
    ) -> Result<(), SimulationError>
    where
        C: ReactionsExtra<Ri, Re>,
        S: SubDomainReactions<Pos, Re, Float>,
        Com: Communicator<
            SubDomainPlainIndex,
            ReactionsExtraBorderReturn<<S as SubDomainReactions<Pos, Re, Float>>::NeighborValue>,
        >,
        Com: Communicator<
            SubDomainPlainIndex,
            ReactionsExtraBorderInfo<<S as SubDomainReactions<Pos, Re, Float>>::BorderInfo>,
        >,
    {
        let mut received_infos = <Com as Communicator<
            SubDomainPlainIndex,
            ReactionsExtraBorderInfo<<S as SubDomainReactions<Pos, Re, Float>>::BorderInfo>,
        >>::receive(&mut self.communicator);
        if determinism {
            received_infos.sort_by_key(|info| info.0);
        }
        for border_info in received_infos {
            let boundary_value = self.subdomain.get_neighbor_value(border_info.1);
            // Send back information about the increment
            self.communicator.send(
                &border_info.0,
                ReactionsExtraBorderReturn(self.subdomain_plain_index, boundary_value),
            )?;
        }
        Ok(())
    }

    /// TODO
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn update_reactions_extra_step_3<Pos, Ri, Re, Float>(
        &mut self,
        determinism: bool,
    ) -> Result<(), SimulationError>
    where
        C: ReactionsExtra<Ri, Re>,
        C: Intracellular<Ri>,
        A: UpdateReactions<Ri>,
        C: Position<Pos>,
        S: SubDomainReactions<Pos, Re, Float>,
        Com: Communicator<
            SubDomainPlainIndex,
            ReactionsExtraBorderReturn<<S as SubDomainReactions<Pos, Re, Float>>::NeighborValue>,
        >,
    {
        let mut errors: Vec<CalcError> = vec![];
        // TODO Think about doing this in two passes
        // Pass 1: Calculate increments (dintra, dextra) and store them in AuxStorage (both!)
        // Pass 2: Use non-mutable iterator (ie. self.voxels.iter()) instead of .iter_mut()
        //         to give results to treat_increments function.
        let sources = self
            .voxels
            .iter_mut()
            .map(|(_, vox)| {
                vox.cells.iter_mut().map(|(cbox, aux_storage)| {
                    let intracellular = cbox.get_intracellular();
                    let pos = cbox.pos();
                    let extracellular = self.subdomain.get_extracellular_at_pos(&pos)?;
                    let (dintra, dextra) =
                        cbox.calculate_combined_increment(&intracellular, &extracellular)?;
                    aux_storage.incr_conc(dintra);
                    Result::<_, CalcError>::Ok((pos, dextra))
                })
            })
            .flatten()
            .collect::<Result<Vec<_>, CalcError>>()?;
        match errors.len() {
            1 => return Err(errors.pop().unwrap().into()),
            _ => (),
        }
        let mut neighbors = <Com as Communicator<
            SubDomainPlainIndex,
            ReactionsExtraBorderReturn<<S as SubDomainReactions<Pos, Re, Float>>::NeighborValue>,
        >>::receive(&mut self.communicator);
        if determinism {
            neighbors.sort_by_key(|v| v.0);
        }
        self.subdomain
            .treat_increments(neighbors.into_iter().map(|n| n.1), sources.into_iter())?;
        Ok(())
    }
}

/// This information will be sent from one cell to another to determine their combined reactions.
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

/// This informatino is returned after receiving [ReactionsContactInformation] and delivers the
/// increment.
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
        C: cellular_raza_concepts::Intracellular<Ri>,
        C: Position<Pos>,
        Ri: cellular_raza_concepts::Xapy<Float>,
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

                aux1.incr_conc(dintra11.xapy(one_half, &dintra21.xa(one_half)));
                aux2.incr_conc(dintra22.xapy(one_half, &dintra12.xa(one_half)));
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
        C: cellular_raza_concepts::Intracellular<Ri>,
        C: Position<Pos>,
        Ri: Xapy<Float>,
        A: UpdateReactions<Ri>,
        A: UpdateReactionsContact<Ri, N>,
        Float: num::Float,
    {
        let one_half = Float::one() / (Float::one() + Float::one());
        let mut dextra_total = ext_intra.xa(Float::zero());
        for (cell, aux_storage) in self.cells.iter_mut() {
            let own_intra = cell.get_intracellular();
            let own_pos = cell.pos();

            let (dintra, dextra) = cell.calculate_contact_increment(
                &own_intra, &ext_intra, &own_pos, &ext_pos, &ext_rinf,
            )?;
            aux_storage.incr_conc(dintra.xa(one_half));
            dextra_total = dextra.xapy(one_half, &dextra_total);
        }
        Ok(dextra_total)
    }
}

impl<I, S, C, A, Com, Sy> SubDomainBox<I, S, C, A, Com, Sy>
where
    S: SubDomain,
{
    /// TODO
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn update_contact_reactions_step_1<Ri, Pos, RInf, Float, const N: usize>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        Pos: Clone,
        C: cellular_raza_concepts::ReactionsContact<Ri, Pos, Float, RInf>,
        C: cellular_raza_concepts::Intracellular<Ri>,
        C: cellular_raza_concepts::Position<Pos>,
        A: UpdateReactions<Ri>,
        A: UpdateReactionsContact<Ri, N>,
        Ri: Xapy<Float> + Clone,
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
                let mut incr = cell_intra.xa(Float::zero());
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
        C: cellular_raza_concepts::Intracellular<Ri>,
        A: UpdateReactions<Ri> + UpdateReactionsContact<Ri, N>,
        Ri: Xapy<Float>,
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
#[cfg_attr(feature = "tracing", instrument(skip_all))]
pub fn local_update_contact_reactions_step_3<
    C,
    A,
    Ri,
    #[cfg(feature = "tracing")] F: core::fmt::Debug,
    #[cfg(not(feature = "tracing"))] F,
>(
    cell: &mut C,
    aux_storage: &mut A,
    _dt: F,
    _rng: &mut rand_chacha::ChaCha8Rng,
) -> Result<(), SimulationError>
where
    A: UpdateReactions<Ri> + UpdateReactionsContact<Ri, 2>,
    C: cellular_raza_concepts::Intracellular<Ri>,
    F: num::Float + FromPrimitive,
    Ri: Xapy<F> + Clone,
{
    reactions_contact_adams_bashforth_3rd(cell, aux_storage)?;
    Ok(())
}

/// TODO
#[allow(private_bounds)]
#[cfg_attr(feature = "tracing", instrument(skip_all))]
pub fn local_reactions_intracellular<
    C,
    A,
    Ri,
    #[cfg(feature = "tracing")] F: core::fmt::Debug,
    #[cfg(not(feature = "tracing"))] F,
    const N: usize,
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
    Ri: Xapy<F>,
    ReactionsRungeKuttaSolver<N>: RungeKutta<N>,
{
    ReactionsRungeKuttaSolver::<N>::update(cell, aux_storage, dt)?;
    Ok(())
}

/// Ensures that intracellular increments have been cleared before the next update step.
#[cfg_attr(feature = "tracing", instrument(skip_all))]
pub fn local_reactions_use_increment<
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
    C: Intracellular<Ri>,
    A: UpdateReactions<Ri>,
    Ri: Xapy<F>,
    F: num::Zero,
{
    let intra = cell.get_intracellular();
    let dintra = aux_storage.get_conc();
    cell.set_intracellular(dintra.xapy(dt, &intra));
    aux_storage.set_conc(aux_storage.get_conc().xa(F::zero()));
    Ok(())
}

/// Performs the increment operation.
///
/// The [SubDomainReactions::update_fluid_dynamics] and [SubDomainReactions::treat_increments] work
/// together as an abstraction to allow for more complicated solvers.
#[cfg_attr(feature = "tracing", instrument(skip_all))]
pub fn local_subdomain_update_reactions_extra<S, Ri, Re, Float>(
    subdomain: &mut S,
    dt: Float,
) -> Result<(), SimulationError>
where
    S: SubDomainReactions<Ri, Re, Float>,
{
    subdomain.update_fluid_dynamics(dt)?;
    Ok(())
}
