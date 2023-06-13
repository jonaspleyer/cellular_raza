use cellular_raza_concepts::cell::*;
use cellular_raza_concepts::cycle::*;
use cellular_raza_concepts::domain::*;
use cellular_raza_concepts::errors::*;
use cellular_raza_concepts::interaction::*;
use cellular_raza_concepts::mechanics::*;

use super::errors::*;
use crate::storage::concepts::{StorageInterface, StorageManager};

use std::collections::{BTreeMap, HashMap};
use std::marker::{Send, Sync};

use std::ops::AddAssign;
use std::ops::{Add, Mul};

use crossbeam_channel::{Receiver, SendError, Sender};
use hurdles::Barrier;

use num::Zero;
use serde::{Deserialize, Serialize};

use rand_chacha::ChaCha8Rng;

#[derive(Clone, Serialize, Deserialize)]
pub struct DomainBox<Dom>
where
    Dom: Serialize + for<'a> Deserialize<'a>,
{
    #[serde(bound = "")]
    pub domain_raw: Dom,
}

impl<Dom> From<Dom> for DomainBox<Dom>
where
    Dom: Serialize + for<'a> Deserialize<'a>,
{
    fn from(domain: Dom) -> DomainBox<Dom> {
        DomainBox { domain_raw: domain }
    }
}

impl<Cel, Ind, Vox, Dom> Domain<CellAgentBox<Cel>, Ind, Vox> for DomainBox<Dom>
where
    Dom: Domain<Cel, Ind, Vox>,
    Vox: Send + Sync,
    Cel: Serialize + for<'a> Deserialize<'a> + Send + Sync,
{
    fn apply_boundary(&self, cbox: &mut CellAgentBox<Cel>) -> Result<(), BoundaryError> {
        self.domain_raw.apply_boundary(&mut cbox.cell)
    }

    fn get_neighbor_voxel_indices(&self, index: &Ind) -> Vec<Ind> {
        self.domain_raw.get_neighbor_voxel_indices(index)
    }

    fn get_voxel_index(&self, cbox: &CellAgentBox<Cel>) -> Ind {
        self.domain_raw.get_voxel_index(&cbox.cell)
    }

    fn get_all_indices(&self) -> Vec<Ind> {
        self.domain_raw.get_all_indices()
    }

    fn generate_contiguous_multi_voxel_regions(
        &self,
        n_regions: usize,
    ) -> Result<(usize, Vec<Vec<(Ind, Vox)>>), CalcError> {
        self.domain_raw
            .generate_contiguous_multi_voxel_regions(n_regions)
    }
}

/// This is a purely implementational detail and should not be of any concern to the end user.
pub type PlainIndex = u64;

pub struct IndexBoundaryInformation<Ind> {
    pub index_original_sender: PlainIndex,
    pub index_original_sender_raw: Ind,
    pub index_original_receiver: PlainIndex,
}

pub struct ConcentrationBoundaryInformation<ConcVecExtracellular, Ind> {
    pub index_original_sender: PlainIndex,
    pub concentration_boundary: BoundaryCondition<ConcVecExtracellular>,
    pub index_original_receiver_raw: Ind,
}

pub struct PosInformation<Pos, Vel, Inf> {
    pub pos: Pos,
    pub vel: Vel,
    pub info: Option<Inf>,
    pub count: usize,
    pub index_sender: PlainIndex,
    pub index_receiver: PlainIndex,
}

pub struct ForceInformation<For> {
    pub force: For,
    pub count: usize,
    pub index_sender: PlainIndex,
}

pub trait GetPlainIndex {
    fn get_plain_index(&self) -> PlainIndex;
}

#[derive(Serialize, Deserialize, Clone)]
pub struct VoxelBox<
    Ind,
    Pos,
    Vel,
    For,
    Vox,
    Cel,
    ConcVecExtracellular,
    ConcBoundaryExtracellular,
    ConcVecIntracellular,
> where
    Pos: Serialize + for<'a> Deserialize<'a>,
    For: Serialize + for<'a> Deserialize<'a>,
    Vel: Serialize + for<'a> Deserialize<'a>,
    Cel: Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcBoundaryExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a>,
{
    pub plain_index: PlainIndex,
    pub index: Ind,
    pub voxel: Vox,
    pub neighbors: Vec<PlainIndex>,
    #[serde(bound = "")]
    pub cells: Vec<(
        CellAgentBox<Cel>,
        AuxiliaryCellPropertyStorage<Pos, Vel, For, ConcVecIntracellular>,
    )>,
    #[serde(bound = "")]
    pub new_cells: Vec<(Cel, Option<CellularIdentifier>)>,
    pub id_counter: u64,
    pub rng: ChaCha8Rng,
    #[serde(bound = "")]
    pub extracellular_concentration_increments: Vec<(Pos, ConcVecExtracellular)>,
    #[serde(bound = "")]
    pub concentration_boundaries: Vec<(Ind, BoundaryCondition<ConcBoundaryExtracellular>)>,
}

impl<
        Ind,
        Vox,
        Cel,
        Pos,
        Vel,
        For,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
    > GetPlainIndex
    for VoxelBox<
        Ind,
        Pos,
        Vel,
        For,
        Vox,
        Cel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
    >
where
    Pos: Serialize + for<'a> Deserialize<'a>,
    For: Serialize + for<'a> Deserialize<'a>,
    Vel: Serialize + for<'a> Deserialize<'a>,
    Cel: Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcBoundaryExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a>,
{
    fn get_plain_index(&self) -> PlainIndex {
        self.plain_index
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AuxiliaryCellPropertyStorage<Pos, Vel, For, ConcVecIntracellular> {
    force: For,
    intracellular_concentration_increment: ConcVecIntracellular,
    pub(crate) cycle_events: Vec<CycleEvent>,

    inc_pos_back_1: Option<Pos>,
    inc_pos_back_2: Option<Pos>,
    inc_vel_back_1: Option<Vel>,
    inc_vel_back_2: Option<Vel>,
}

impl<Pos, Vel, For, ConcVecIntracellular> Default
    for AuxiliaryCellPropertyStorage<Pos, Vel, For, ConcVecIntracellular>
where
    For: Zero,
    ConcVecIntracellular: Zero,
{
    fn default() -> AuxiliaryCellPropertyStorage<Pos, Vel, For, ConcVecIntracellular> {
        AuxiliaryCellPropertyStorage {
            force: For::zero(),
            intracellular_concentration_increment: ConcVecIntracellular::zero(),
            cycle_events: Vec::new(),

            inc_pos_back_1: None,
            inc_pos_back_2: None,
            inc_vel_back_1: None,
            inc_vel_back_2: None,
        }
    }
}

impl<
        Ind,
        Vox,
        Cel,
        Pos,
        Vel,
        For,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
    >
    VoxelBox<
        Ind,
        Pos,
        Vel,
        For,
        Vox,
        Cel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
    >
where
    Ind: Clone,
    Pos: Serialize + for<'a> Deserialize<'a>,
    For: num::Zero + Serialize + for<'a> Deserialize<'a>,
    Vel: Serialize + for<'a> Deserialize<'a>,
    Cel: Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcBoundaryExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a> + Zero,
{
    pub fn new(
        plain_index: PlainIndex,
        index: Ind,
        voxel: Vox,
        neighbors: Vec<PlainIndex>,
        cells: Vec<CellAgentBox<Cel>>,
    ) -> VoxelBox<
        Ind,
        Pos,
        Vel,
        For,
        Vox,
        Cel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
    > {
        use rand::SeedableRng;
        let n_cells = cells.len() as u64;
        VoxelBox {
            plain_index,
            index,
            voxel,
            neighbors,
            cells: cells
                .into_iter()
                .map(|cell| (cell, AuxiliaryCellPropertyStorage::default()))
                .collect(),
            new_cells: Vec::new(),
            id_counter: n_cells,
            rng: ChaCha8Rng::seed_from_u64(plain_index as u64 * 10),
            extracellular_concentration_increments: Vec::new(),
            concentration_boundaries: Vec::new(),
        }
    }
}

impl<
        Ind,
        Vox,
        Cel,
        Pos,
        Vel,
        For,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
    >
    VoxelBox<
        Ind,
        Pos,
        Vel,
        For,
        Vox,
        Cel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
    >
where
    Ind: Clone,
    Pos: Serialize + for<'a> Deserialize<'a>,
    For: Serialize + for<'a> Deserialize<'a>,
    Vel: Serialize + for<'a> Deserialize<'a>,
    Cel: Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcBoundaryExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a>,
{
    fn calculate_custom_force_on_cells(&mut self) -> Result<(), CalcError>
    where
        Vox: Voxel<Ind, Pos, Vel, For>,
        Ind: Index,
        Pos: Position,
        Vel: Velocity,
        For: Force,
        Cel: Mechanics<Pos, Vel, For>,
    {
        for (cell, aux_storage) in self.cells.iter_mut() {
            match self
                .voxel
                .custom_force_on_cell(&cell.pos(), &cell.velocity())
            {
                Some(Ok(force)) => Ok(aux_storage.force += force),
                Some(Err(e)) => Err(e),
                None => Ok(()),
            }?;
        }
        Ok(())
    }

    fn calculate_force_between_cells_internally<Inf>(&mut self) -> Result<(), CalcError>
    where
        Vox: Voxel<Ind, Pos, Vel, For>,
        Ind: Index,
        Pos: Position,
        Vel: Velocity,
        For: Force,
        Cel: Interaction<Pos, Vel, For, Inf> + Mechanics<Pos, Vel, For> + Clone,
    {
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
                    aux1.force -= force.clone() * 0.5;
                    aux2.force += force * 0.5;
                }

                if let Some(force_result) = c2.calculate_force_between(&p2, &v2, &p1, &v1, &i1) {
                    let force = force_result?;
                    aux1.force -= force.clone() * 0.5;
                    aux2.force += force * 0.5;
                }
            }
        }
        Ok(())
    }

    fn calculate_force_between_cells_external<Inf>(
        &mut self,
        ext_pos: &Pos,
        ext_vel: &Vel,
        ext_inf: &Option<Inf>,
    ) -> Result<For, CalcError>
    where
        Vox: Voxel<Ind, Pos, Vel, For>,
        Ind: Index,
        Pos: Position,
        Vel: Velocity,
        For: Force,
        Cel: Interaction<Pos, Vel, For, Inf> + Mechanics<Pos, Vel, For>,
    {
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
                    aux_storage.force -= f.clone() * 0.5;
                    force += f * 0.5;
                }
                Some(Err(e)) => return Err(e),
                None => (),
            };
        }
        Ok(force)
    }

    fn update_cell_cycle(&mut self, dt: &f64) -> Result<(), SimulationError>
    where
        Cel: Cycle<Cel>,
        AuxiliaryCellPropertyStorage<Pos, Vel, For, ConcVecIntracellular>: Default,
    {
        // Update the cell individual cells
        self.cells
            .iter_mut()
            .map(|(cbox, aux_storage)| {
                // Check for cycle events and do update if necessary
                let mut remaining_events = Vec::new();
                for event in aux_storage.cycle_events.drain(..) {
                    match event {
                        CycleEvent::Division => {
                            let new_cell = Cel::divide(&mut self.rng, &mut cbox.cell)?;
                            self.new_cells.push((new_cell, Some(cbox.get_id())));
                        }
                        CycleEvent::Remove => remaining_events.push(event),
                        CycleEvent::PhasedDeath => {
                            remaining_events.push(event);
                        }
                    };
                }
                aux_storage.cycle_events = remaining_events;
                // Update the cell cycle
                if aux_storage.cycle_events.contains(&CycleEvent::PhasedDeath) {
                    match Cel::update_conditional_phased_death(&mut self.rng, dt, &mut cbox.cell)? {
                        true => aux_storage.cycle_events.push(CycleEvent::Remove),
                        false => (),
                    }
                } else {
                    match Cel::update_cycle(&mut self.rng, dt, &mut cbox.cell) {
                        Some(event) => aux_storage.cycle_events.push(event),
                        None => (),
                    }
                }
                Ok(())
            })
            .collect::<Result<(), SimulationError>>()?;

        // Remove cells which are flagged for death
        self.cells
            .retain(|(_, aux_storage)| !aux_storage.cycle_events.contains(&CycleEvent::Remove));

        // Include new cells
        self.cells
            .extend(self.new_cells.drain(..).map(|(cell, parent_id)| {
                self.id_counter += 1;
                (
                    CellAgentBox::new(self.plain_index, self.id_counter, cell, parent_id),
                    AuxiliaryCellPropertyStorage::default(),
                )
            }));
        Ok(())
    }
}

/* impl<I,V,C,Pos,For> Voxel<PlainIndex,Pos,For> for VoxelBox<I, V,C,For>
where
    Cel: Clone + Serialize + for<'a> Deserialize<'a> + Send + Sync,
    Pos: Serialize + for<'a> Deserialize<'a> + Send + Sync,
    For: Clone + Serialize + for<'a> Deserialize<'a> + Send + Sync,
    Ind: Serialize + for<'a> Deserialize<'a> + Index,
    V: Serialize + for<'a> Deserialize<'a> + Voxel<Ind,Pos,For>,
{
    fn custom_force_on_cell(&self, cell: &Pos) -> Option<Result<For, CalcError>> {
        self.voxel.custom_force_on_cell(cell)
    }

    fn get_index(&self) -> PlainIndex {
        self.plain_index
    }
}*/

// This object has multiple voxels and runs on a single thread.
// It can communicate with other containers via channels.
pub struct MultiVoxelContainer<
    Ind,
    Pos,
    Vel,
    For,
    Inf,
    Vox,
    Dom,
    Cel,
    ConcVecExtracellular = (),
    ConcBoundaryExtracellular = (),
    ConcVecIntracellular = (),
> where
    Pos: Serialize + for<'a> Deserialize<'a>,
    For: Serialize + for<'a> Deserialize<'a>,
    Vel: Serialize + for<'a> Deserialize<'a>,
    Cel: Serialize + for<'a> Deserialize<'a>,
    Dom: Serialize + for<'a> Deserialize<'a>,
    ConcVecExtracellular: Serialize + for<'a> Deserialize<'a> + 'static,
    ConcBoundaryExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a>,
{
    pub voxels: BTreeMap<
        PlainIndex,
        VoxelBox<
            Ind,
            Pos,
            Vel,
            For,
            Vox,
            Cel,
            ConcVecExtracellular,
            ConcBoundaryExtracellular,
            ConcVecIntracellular,
        >,
    >,

    // TODO
    // Maybe we need to implement this somewhere else since
    // it is currently not simple to change this variable on the fly.
    // However, maybe we should be thinking about specifying an interface to use this function
    // Something like:
    // fn update_domain(&mut self, domain: Domain) -> Result<(), BoundaryError>
    // And then automatically have the ability to change cell positions if the domain shrinks/grows for example
    // but then we might also want to change the number of voxels and redistribute cells accordingly
    // This needs much more though!
    pub domain: DomainBox<Dom>,
    pub index_to_plain_index: BTreeMap<Ind, PlainIndex>,
    pub plain_index_to_thread: BTreeMap<PlainIndex, usize>,
    pub index_to_thread: BTreeMap<Ind, usize>,

    pub senders_cell: HashMap<usize, Sender<CellAgentBox<Cel>>>,
    pub senders_pos: HashMap<usize, Sender<PosInformation<Pos, Vel, Inf>>>,
    pub senders_force: HashMap<usize, Sender<ForceInformation<For>>>,

    pub senders_boundary_index: HashMap<usize, Sender<IndexBoundaryInformation<Ind>>>,
    pub senders_boundary_concentrations:
        HashMap<usize, Sender<ConcentrationBoundaryInformation<ConcBoundaryExtracellular, Ind>>>,

    // Same for receiving
    pub receiver_cell: Receiver<CellAgentBox<Cel>>,
    pub receiver_pos: Receiver<PosInformation<Pos, Vel, Inf>>,
    pub receiver_force: Receiver<ForceInformation<For>>,

    pub receiver_index: Receiver<IndexBoundaryInformation<Ind>>,
    pub receiver_concentrations:
        Receiver<ConcentrationBoundaryInformation<ConcBoundaryExtracellular, Ind>>,

    // Global barrier to synchronize threads and make sure every information is sent before further processing
    pub barrier: Barrier,

    pub storage_cells: StorageManager<CellularIdentifier, CellAgentBox<Cel>>,
    pub storage_voxels: StorageManager<
        PlainIndex,
        VoxelBox<
            Ind,
            Pos,
            Vel,
            For,
            Vox,
            Cel,
            ConcVecExtracellular,
            ConcBoundaryExtracellular,
            ConcVecIntracellular,
        >,
    >,

    pub mvc_id: u32,
}

impl<
        Ind,
        Pos,
        Vel,
        For,
        Inf,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
        Vox,
        Dom,
        Cel,
    >
    MultiVoxelContainer<
        Ind,
        Pos,
        Vel,
        For,
        Inf,
        Vox,
        Dom,
        Cel,
        ConcVecExtracellular,
        ConcBoundaryExtracellular,
        ConcVecIntracellular,
    >
where
    // TODO abstract away these trait bounds to more abstract traits
    // these traits should be defined when specifying the individual cell components
    // (eg. mechanics, interaction, etc...)
    Ind: Index + Serialize + for<'a> Deserialize<'a>,
    Vox: Voxel<Ind, Pos, Vel, For>,
    Dom: Domain<Cel, Ind, Vox>,
    Pos: Serialize + for<'a> Deserialize<'a>,
    Vel: Serialize + for<'a> Deserialize<'a>,
    For: Force + Serialize + for<'a> Deserialize<'a>,
    Inf: Clone,
    Cel: Serialize + for<'a> Deserialize<'a> + Send + Sync,
    ConcVecExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcBoundaryExtracellular: Serialize + for<'a> Deserialize<'a>,
    ConcVecIntracellular: Serialize + for<'a> Deserialize<'a> + Zero,
{
    fn update_local_functions<ConcGradientExtracellular, ConcTotalExtracellular>(
        &mut self,
        dt: &f64,
    ) -> Result<(), SimulationError>
    where
        Pos: Position,
        Vel: Velocity,
        For: Force,
        Inf: InteractionInformation,
        Cel: CellAgent<Pos, Vel, For, Inf>
            + InteractionExtracellularGradient<Cel, ConcGradientExtracellular>,
        Vox: ExtracellularMechanics<
            Ind,
            Pos,
            ConcVecExtracellular,
            ConcGradientExtracellular,
            ConcTotalExtracellular,
            ConcBoundaryExtracellular,
        >,
    {
        self.voxels
            .iter_mut()
            .map(|(_, vox)| {
                // Update all local functions inside the voxel
                vox.update_cell_cycle(dt)?;

                // TODO every voxel should apply its own boundary conditions
                // This is now a global rule but we do not want this
                // This should not be dependent on the domain
                // Apply boundary conditions to the cells in the respective voxels
                vox.cells
                    .iter_mut()
                    .map(|(cell, _)| self.domain.apply_boundary(cell))
                    .collect::<Result<(), BoundaryError>>()?;

                #[cfg(feature = "gradients")]
                vox.cells
                    .iter_mut()
                    .map(|(cell, _)| {
                        let gradient = vox
                            .voxel
                            .get_extracellular_gradient_at_point(&cell.cell.pos())?;
                        Cel::sense_gradient(&mut cell.cell, &gradient)?;
                        Ok(())
                    })
                    .collect::<Result<(), SimulationError>>()?;
                Ok(())
            })
            .collect::<Result<(), SimulationError>>()
    }

    // TODO make sure that if no extracellular mechanics are in action updating is correct and the trait may be adjusted
    fn update_cellular_reactions<ConcGradientExtracellular, ConcTotalExtracellular>(
        &mut self,
        dt: &f64,
    ) -> Result<(), SimulationError>
    where
        ConcVecIntracellular: std::ops::AddAssign + Mul<f64, Output = ConcVecIntracellular>,
        Cel: Mechanics<Pos, Vel, For>
            + CellularReactions<ConcVecIntracellular, ConcVecExtracellular>,
        Vox: ExtracellularMechanics<
            Ind,
            Pos,
            ConcVecExtracellular,
            ConcGradientExtracellular,
            ConcTotalExtracellular,
            ConcBoundaryExtracellular,
        >,
    {
        self.voxels
            .iter_mut()
            .map(|(_, voxelbox)| {
                voxelbox.cells.iter_mut().map(
                    |(cellbox, _aux_storage)| -> Result<(), SimulationError> {
                        let internal_concentration_vector = cellbox.cell.get_intracellular();
                        #[cfg(feature = "fluid_mechanics")]
                        let external_concentration_vector = voxelbox
                            .voxel
                            .get_extracellular_at_point(&cellbox.cell.pos())?;
                        let (increment_intracellular, increment_extracellular) = cellbox
                            .cell
                            .calculate_intra_and_extracellular_reaction_increment(
                                &internal_concentration_vector,
                                #[cfg(feature = "fluid_mechanics")]
                                &external_concentration_vector,
                            )?;
                        // aux_storage.intracellular_concentration_increment += increment_intracellular;
                        voxelbox
                            .extracellular_concentration_increments
                            .push((cellbox.cell.pos(), increment_extracellular));
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
            .collect::<Result<(), SimulationError>>()
    }

    pub fn sort_cell_in_voxel(&mut self, cell: CellAgentBox<Cel>) -> Result<(), SimulationError> {
        let index = self.index_to_plain_index[&self.domain.get_voxel_index(&cell)];
        let aux_storage = AuxiliaryCellPropertyStorage::default();

        match self.voxels.get_mut(&index) {
            Some(vox) => vox.cells.push((cell, aux_storage)),
            None => {
                let thread_index = self.plain_index_to_thread[&index];
                match self.senders_cell.get(&thread_index) {
                    Some(sender) => sender.send(cell),
                    None => Err(SendError(cell)),
                }?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "fluid_mechanics")]
    pub fn update_fluid_mechanics_step_1<ConcGradientExtracellular, ConcTotalExtracellular>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        Vox: ExtracellularMechanics<
            Ind,
            Pos,
            ConcVecExtracellular,
            ConcGradientExtracellular,
            ConcTotalExtracellular,
            ConcBoundaryExtracellular,
        >,
    {
        let indices_iterator = self
            .voxels
            .iter()
            .map(|(ind, vox)| (ind.clone(), vox.index.clone(), vox.neighbors.clone()))
            .collect::<Vec<_>>();
        for (voxel_plain_index, voxel_index, neighbor_indices) in indices_iterator.into_iter() {
            for neighbor_index in neighbor_indices {
                match self.voxels.get(&neighbor_index) {
                    Some(neighbor_voxel) => {
                        let neighbor_voxel_index_raw = neighbor_voxel.index.clone();
                        let bc = neighbor_voxel
                            .voxel
                            .boundary_condition_to_neighbor_voxel(&voxel_index)?;
                        let vox = self.voxels.get_mut(&voxel_plain_index).unwrap();
                        vox.concentration_boundaries
                            .push((neighbor_voxel_index_raw, bc));
                        Ok(())
                    }
                    None => self.senders_boundary_index
                        [&self.plain_index_to_thread[&neighbor_index]]
                        .send(IndexBoundaryInformation {
                            index_original_sender: voxel_plain_index,
                            index_original_receiver: neighbor_index,
                            index_original_sender_raw: voxel_index.clone(),
                        }),
                }?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "fluid_mechanics")]
    pub fn update_fluid_mechanics_step_2<ConcGradientExtracellular, ConcTotalExtracellular>(
        &mut self,
    ) -> Result<(), SimulationError>
    where
        Vox: ExtracellularMechanics<
            Ind,
            Pos,
            ConcVecExtracellular,
            ConcGradientExtracellular,
            ConcTotalExtracellular,
            ConcBoundaryExtracellular,
        >,
    {
        // Receive IndexBoundaryInformation and send back BoundaryConcentrationInformation
        for index_boundary_information in self.receiver_index.try_iter() {
            let voxel_box = self
                .voxels
                .get(&index_boundary_information.index_original_receiver)
                .ok_or(IndexError {
                    message: format!(""),
                })?;

            // Obtain the boundary concentrations to this voxel
            let concentration_boundary = voxel_box.voxel.boundary_condition_to_neighbor_voxel(
                &index_boundary_information.index_original_sender_raw,
            )?;

            // Send back the concentrations here
            let thread_index =
                self.plain_index_to_thread[&index_boundary_information.index_original_sender];
            self.senders_boundary_concentrations[&thread_index].send(
                ConcentrationBoundaryInformation {
                    index_original_sender: index_boundary_information.index_original_sender,
                    index_original_receiver_raw: voxel_box.voxel.get_index(),
                    concentration_boundary,
                },
            )?;
        }

        Ok(())
    }

    #[cfg(feature = "fluid_mechanics")]
    pub fn update_fluid_mechanics_step_3<ConcGradientExtracellular, ConcTotalExtracellular>(
        &mut self,
        dt: &f64,
    ) -> Result<(), SimulationError>
    where
        ConcVecExtracellular: Concentration,
        ConcTotalExtracellular: Concentration,
        Vox: ExtracellularMechanics<
            Ind,
            Pos,
            ConcVecExtracellular,
            ConcGradientExtracellular,
            ConcTotalExtracellular,
            ConcBoundaryExtracellular,
        >,
    {
        // Update boundary conditions with new
        for concentration_boundary_information in self.receiver_concentrations.try_iter() {
            let vox = self.voxels.get_mut(&concentration_boundary_information.index_original_sender).ok_or(IndexError { message: format!("EngineError: Sender with plain index {} was ended up in location where index is not present anymore", concentration_boundary_information.index_original_sender)})?;
            vox.concentration_boundaries.push((
                concentration_boundary_information.index_original_receiver_raw,
                concentration_boundary_information.concentration_boundary,
            ));
        }

        self.voxels
            .iter_mut()
            .map(|(_, voxel_box)| -> Result<(), SimulationError> {
                let total_extracellular = voxel_box.voxel.get_total_extracellular();
                let concentration_increment = voxel_box.voxel.calculate_increment(
                    &total_extracellular,
                    &voxel_box.extracellular_concentration_increments,
                    &voxel_box.concentration_boundaries[..],
                )?;
                // Update the gradients before we set new extracllular because otherwise it would be inaccurate. This way the gradients are "behind" the actual concentrations by one timestep
                #[cfg(feature = "gradients")]
                voxel_box
                    .voxel
                    .update_extracellular_gradient(&voxel_box.concentration_boundaries[..])?;
                voxel_box.voxel.set_total_extracellular(
                    &(total_extracellular + concentration_increment * *dt),
                )?;
                voxel_box.extracellular_concentration_increments.drain(..);
                voxel_box.concentration_boundaries.drain(..);
                Ok(())
            })
            .collect::<Result<(), SimulationError>>()?;
        Ok(())
    }

    // TODO the trait bounds here are too harsh. We should not be required to have Pos: Position or Vel: Velocity here at all!
    pub fn update_cellular_mechanics_step_1(&mut self) -> Result<(), SimulationError>
    where
        Pos: Position,
        Vel: Velocity,
        Inf: Clone,
        For: std::fmt::Debug,
        Cel: Interaction<Pos, Vel, For, Inf> + Mechanics<Pos, Vel, For> + Clone,
    {
        // General Idea of this function
        // for each cell
        //      for each neighbor_voxel in neighbors of voxel containing cell
        //              if neighbor_voxel is in current MultivoxelContainer
        //                      calculate forces of current cells on cell and store
        //                      calculate force from voxel on cell and store
        //              else
        //                      send PosInformation to other MultivoxelContainer
        //
        // for each PosInformation received from other MultivoxelContainers
        //      calculate forces of current_cells on cell and send back
        //
        // for each ForceInformation received from other MultivoxelContainers
        //      store received force
        //
        // for each cell in this MultiVoxelContainer
        //      update pos and velocity with all forces obtained
        //      Simultanously

        // Calculate forces between cells of own voxel
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
                        None => Ok(
                            self.senders_pos[&self.plain_index_to_thread[&neighbor_index]].send(
                                PosInformation {
                                    index_sender: voxel_index,
                                    index_receiver: neighbor_index.clone(),
                                    pos: cell_pos.clone(),
                                    vel: cell_vel.clone(),
                                    info: cell_inf.clone(),
                                    count: cell_count,
                                },
                            )?,
                        ),
                    }?;
                }
                self.voxels.get_mut(&voxel_index).unwrap().cells[cell_count]
                    .1
                    .force += force;
            }
        }

        // Calculate custom force of voxel on cell
        self.voxels
            .iter_mut()
            .map(|(_, vox)| vox.calculate_custom_force_on_cells())
            .collect::<Result<(), CalcError>>()?;
        Ok(())
    }

    pub fn update_cellular_mechanics_step_2(&mut self) -> Result<(), SimulationError>
    where
        Pos: Position,
        Vel: Velocity,
        Cel: Interaction<Pos, Vel, For, Inf> + Mechanics<Pos, Vel, For> + Clone,
    {
        // Receive PositionInformation and send back ForceInformation
        for pos_info in self.receiver_pos.try_iter() {
            let vox = self.voxels.get_mut(&pos_info.index_receiver).ok_or(IndexError {message: format!("EngineError: Voxel with index {:?} of PosInformation can not be found in this thread.", pos_info.index_receiver)})?;
            // Calculate force from cells in voxel
            let force = vox.calculate_force_between_cells_external(
                &pos_info.pos,
                &pos_info.vel,
                &pos_info.info,
            )?;

            // Send back force information
            let thread_index = self.plain_index_to_thread[&pos_info.index_sender];
            self.senders_force[&thread_index].send(ForceInformation {
                force,
                count: pos_info.count,
                index_sender: pos_info.index_sender,
            })?;
        }
        Ok(())
    }

    pub fn update_cellular_mechanics_step_3(&mut self, dt: &f64) -> Result<(), SimulationError>
    where
        Pos: Position,
        Vel: Velocity,
        Cel: Interaction<Pos, Vel, For, Inf> + Mechanics<Pos, Vel, For> + Clone,
    {
        // Update position and velocity of all cells with new information
        for obt_forces in self.receiver_force.try_iter() {
            let vox = self.voxels.get_mut(&obt_forces.index_sender).ok_or(IndexError { message: format!("EngineError: Sender with plain index {} was ended up in location where index is not present anymore", obt_forces.index_sender)})?;
            match vox.cells.get_mut(obt_forces.count) {
                Some((_, aux_storage)) => Ok(aux_storage.force+=obt_forces.force),
                None => Err(IndexError { message: format!("EngineError: Force Information with sender index {:?} and cell at vector position {} could not be matched", obt_forces.index_sender, obt_forces.count)}),
            }?;
        }

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
        }
        Ok(())
    }

    pub fn sort_cells_in_voxels_step_1(&mut self) -> Result<(), SimulationError>
    where
        Pos: Position,
        Vel: Velocity,
        Cel: Mechanics<Pos, Vel, For>,
    {
        // Store all cells which need to find a new home in this variable
        let mut find_new_home_cells = Vec::<_>::new();

        for (voxel_index, vox) in self.voxels.iter_mut() {
            // Drain every cell which is currently not in the correct voxel
            // TODO use drain_filter when stabilized
            let (new_voxel_cells, old_voxel_cells): (Vec<_>, Vec<_>) =
                vox.cells.drain(..).partition(|(c, _)| {
                    match self
                        .index_to_plain_index
                        .get(&self.domain.get_voxel_index(&c))
                    {
                        Some(ind) => ind != voxel_index,
                        None => panic!("Cannot find index {:?}", self.domain.get_voxel_index(&c)),
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
            let ind = self.domain.get_voxel_index(&cell);
            let new_thread_index = self.index_to_thread[&ind];
            let cell_index = self.index_to_plain_index[&ind];
            match self.voxels.get_mut(&cell_index) {
                // If new voxel is in current multivoxelcontainer then save them there
                Some(vox) => {
                    vox.cells.push((cell, aux_storage));
                    Ok(())
                }
                // Otherwise send them to the correct other multivoxelcontainer
                None => match self.senders_cell.get(&new_thread_index) {
                    Some(sender) => {
                        sender.send(cell)?;
                        Ok(())
                    }
                    None => Err(IndexError {
                        message: format!(
                            "Could not correctly send cell with id {:?}",
                            cell.get_id()
                        ),
                    }),
                },
            }?;
        }
        Ok(())
    }

    pub fn sort_cells_in_voxels_step_2(&mut self) -> Result<(), SimulationError> {
        // Now receive new cells and insert them
        let mut new_cells = self.receiver_cell.try_iter().collect::<Vec<_>>();
        for cell in new_cells.drain(..) {
            self.sort_cell_in_voxel(cell)?;
        }
        Ok(())
    }

    pub fn save_cells_to_database(&self, iteration: &u64) -> Result<(), StorageError>
    where
        Cel: 'static,
        CellAgentBox<Cel>: Clone,
        AuxiliaryCellPropertyStorage<Pos, Vel, For, ConcVecIntracellular>: Clone,
    {
        let cells = self
            .voxels
            .iter()
            .map(|(_, vox)| vox.cells.clone().into_iter().map(|(c, _)| (c.get_id(), c)))
            .flatten()
            .collect::<Vec<_>>();

        self.storage_cells.store_batch_elements(*iteration, &cells)
    }

    pub fn save_voxels_to_database(&self, iteration: &u64) -> Result<(), StorageError>
    where
        VoxelBox<
            Ind,
            Pos,
            Vel,
            For,
            Vox,
            Cel,
            ConcVecExtracellular,
            ConcBoundaryExtracellular,
            ConcVecIntracellular,
        >: Clone + Send + Sync + 'static,
    {
        let voxels = self
            .voxels
            .clone()
            .into_iter()
            .map(|(_, voxel)| (voxel.get_plain_index(), voxel))
            .collect::<Vec<_>>();

        self.storage_voxels
            .store_batch_elements(*iteration, &voxels)
    }

    // TODO find better function signature to have multiple time-scales
    // or split into different functions
    pub fn run_full_update<ConcGradientExtracellular, ConcTotalExtracellular>(
        &mut self,
        dt: &f64,
    ) -> Result<(), SimulationError>
    where
        Inf: Send + Sync + core::fmt::Debug,
        Pos: Position,
        Vel: Velocity,
        ConcVecExtracellular: Concentration,
        ConcTotalExtracellular: Concentration,
        ConcVecIntracellular: Mul<f64, Output = ConcVecIntracellular>
            + Add<ConcVecIntracellular, Output = ConcVecIntracellular>
            + AddAssign<ConcVecIntracellular>,
        Cel: Cycle<Cel>
            + Mechanics<Pos, Vel, For>
            + Interaction<Pos, Vel, For, Inf>
            + CellularReactions<ConcVecIntracellular, ConcVecExtracellular>
            + InteractionExtracellularGradient<Cel, ConcGradientExtracellular>
            + Clone,
        Vox: ExtracellularMechanics<
            Ind,
            Pos,
            ConcVecExtracellular,
            ConcGradientExtracellular,
            ConcTotalExtracellular,
            ConcBoundaryExtracellular,
        >,
    {
        // These methods are used for sending requests and gathering information in general
        // This gathers information of forces acting between cells and send between threads
        #[cfg(feature = "fluid_mechanics")]
        self.update_fluid_mechanics_step_1()?;

        // Gather boundary conditions between voxels and domain boundaries and send between threads
        self.update_cellular_mechanics_step_1()?;

        // Wait for all threads to synchronize.
        // The goal is to have as few as possible synchronizations
        self.barrier.wait();

        #[cfg(feature = "fluid_mechanics")]
        self.update_fluid_mechanics_step_2()?;

        self.update_cellular_mechanics_step_2()?;

        self.barrier.wait();

        self.update_cellular_reactions(dt)?;

        // These are the true update steps where cell agents are modified the order here may play a role!
        #[cfg(feature = "fluid_mechanics")]
        self.update_fluid_mechanics_step_3(dt)?;

        self.update_cellular_mechanics_step_3(dt)?;

        // TODO this currently also does application of domain boundaries and inclusion of new cells which is wrong in general!
        self.update_local_functions(dt)?;

        // This function needs an additional synchronization step which cannot correctly be done in between the other ones
        self.sort_cells_in_voxels_step_1()?;

        self.barrier.wait();

        self.sort_cells_in_voxels_step_2()?;
        Ok(())
    }
}
