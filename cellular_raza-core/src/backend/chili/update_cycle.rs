use super::{CellBox, CellIdentifier, SimulationError, SubDomainBox, UpdateCycle, Voxel};
use cellular_raza_concepts::{domain_new::SubDomain, CalcError};

pub use cellular_raza_concepts::CycleEvent;

#[cfg(feature = "tracing")]
use tracing::instrument;

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
            + cellular_raza_concepts::Id<Identifier = CellIdentifier>,
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

impl<I, S, C, A, Com, Sy> SubDomainBox<I, S, C, A, Com, Sy>
where
    S: SubDomain,
{
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
            + cellular_raza_concepts::Id<Identifier = CellIdentifier>,
        A: UpdateCycle + Default,
    {
        self.voxels
            .iter_mut()
            .map(|(_, vox)| vox.update_cell_cycle_3(dt))
            .collect::<Result<(), SimulationError>>()?;
        Ok(())
    }
}
