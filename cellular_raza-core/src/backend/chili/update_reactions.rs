use super::{SimulationError, SubDomainBox};
// TODO use cellular_raza_concepts::reactions_new::CellularReactions;
use cellular_raza_concepts::SubDomain;

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
