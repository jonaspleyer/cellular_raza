use super::{CellBox, SimulationError, SubDomainBox, UpdateCycle, Voxel};
use cellular_raza_concepts::SubDomain;

pub use cellular_raza_concepts::CycleEvent;

#[cfg(feature = "tracing")]
use tracing::instrument;

impl<C, A> Voxel<C, A> {
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub(crate) fn update_cell_cycle_4<
        #[cfg(feature = "tracing")] Float: core::fmt::Debug,
        #[cfg(not(feature = "tracing"))] Float,
    >(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        C: cellular_raza_concepts::Cycle<C, Float>,
        A: UpdateCycle + Default,
    {
        use cellular_raza_concepts::Id;
        // Update the cell individual cells
        self.cells
            .iter_mut()
            .map(|(cbox, aux_storage)| {
                // Check for cycle events and take action if necessary
                let mut remaining_events = Vec::new();
                for event in aux_storage.drain_cycle_events() {
                    match event {
                        CycleEvent::Division => {
                            let new_cell = C::divide(&mut self.rng, &mut cbox.cell)?;
                            self.id_counter += 1;
                            cbox.identifier.1 = self.id_counter;
                            cbox.identifier.0 = self.plain_index;
                            self.new_cells.push((new_cell, Some(cbox.get_id())));
                        }
                        CycleEvent::Remove => remaining_events.push(event),
                        CycleEvent::PhasedDeath => {
                            remaining_events.push(event);
                        }
                    };
                }
                aux_storage.set_cycle_events(remaining_events);
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
    /// Separate function to update the cell cycle
    ///
    /// Instead of running one big update function for all local rules, we have to treat this cell
    /// cycle differently since new cells could be generated and thus have consequences for other
    /// update steps as well.
    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    pub fn update_cell_cycle_4<
        #[cfg(feature = "tracing")] F: core::fmt::Debug,
        #[cfg(not(feature = "tracing"))] F,
    >(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        C: cellular_raza_concepts::Cycle<C, F>,
        A: UpdateCycle + Default,
    {
        self.voxels
            .iter_mut()
            .map(|(_, vox)| vox.update_cell_cycle_4())
            .collect::<Result<(), SimulationError>>()?;
        Ok(())
    }
}

/// Advances the cycle of a cell by a small time increment `dt`.
pub fn local_cycle_update<C, A, Float>(
    cell: &mut C,
    aux_storage: &mut A,
    dt: Float,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> Result<(), cellular_raza_concepts::DeathError>
where
    C: cellular_raza_concepts::Cycle<C, Float>,
    A: UpdateCycle,
{
    // Update the cell cycle
    if aux_storage
        .get_cycle_events()
        .contains(&CycleEvent::PhasedDeath)
    {
        if C::update_conditional_phased_death(rng, &dt, cell)? {
            aux_storage.add_cycle_event(CycleEvent::Remove);
        }
    } else {
        if let Some(event) = C::update_cycle(rng, &dt, cell) {
            aux_storage.add_cycle_event(event);
        }
    }
    Ok(())
}
