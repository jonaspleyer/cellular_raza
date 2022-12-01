use crate::concepts::errors::*;

use std::collections::{HashMap,VecDeque};
use core::hash::Hash;
use core::cmp::Eq;

use crossbeam_channel::{Sender,Receiver};
use hurdles::Barrier;


pub trait Domain<Cell,Index> {// <Cell, Index, Voxel> {
    fn apply_boundary(&self, cell: &mut Cell) -> Result<(), BoundaryError>;
    // fn generate_voxels(&self) -> HashMap<Index,Voxel>;
    // fn get_neighbor_voxel_indices(&self, index: &Index) -> Vec<Index>;
    fn get_voxel_index(&self, cell: &Cell) -> Index;
}


pub trait Voxel {
    fn calculate_voxel_force_on_cell<Pos,Force>(&self, cell: &Pos) -> Option<Result<Force, CalcError>>;
    fn get_index<Index:Hash>(&self) -> Index;
}


// This object has multiple voxels and runs on a single thread.
// It can communicate with other containers via channels.
pub struct MultiVoxelContainer<Index: Hash, Cell, Pos, Force, V:Voxel, D:Domain<Cell,Index>> {
    voxels: HashMap<Index, V>,
    voxel_cells: HashMap<Index, VecDeque<Cell>>,
    domain: D,
    senders_cell: HashMap<Index, Sender<Cell>>,
    senders_pos: HashMap<Index, Sender<Pos>>,
    senders_force: HashMap<Index, Sender<Force>>,
    receiver_cell: Receiver<Cell>,
    receiver_pos: Receiver<Pos>,
    receiver_force: Receiver<Force>,
    sending_barrier: Barrier,
}


impl<Index: Hash+Eq, Cell, Pos, Force, V: Voxel, D: Domain<Cell,Index>> MultiVoxelContainer<Index, Cell, Pos, Force, V, D> {
    fn update_cell_cycle(&mut self, dt: f64) {
        self.voxel_cells
            .iter_mut()
            .for_each(|(_, cs)| cs
                .iter_mut()
                .for_each(|c| println!("Updated cycle"))
            );
    }

    fn apply_boundaries(&mut self, domain: &D) -> Result<(), BoundaryError> {
        self.voxel_cells
            .iter_mut()
            .map(|(_, cells)| cells.iter_mut())
            .flatten()
            .map(|cell| domain.apply_boundary(cell))
            .collect()
    }

    fn insert_cells(&mut self, index: Index, new_cells: &mut VecDeque<Cell>) -> Result<(), CalcError> {
        self.voxel_cells.get_mut(&index)
            .ok_or(CalcError{ message: "New cell has incorrect index".to_owned()})?
            .append(new_cells);
        Ok(())
    }
}

// These should be sharable between threads
unsafe impl<Index: Hash, Cell, Pos, Force, V:Voxel, D:Domain<Cell,Index>> Send for MultiVoxelContainer<Index, Cell, Pos, Force, V, D> {}
unsafe impl<Index: Hash, Cell, Pos, Force, V:Voxel, D:Domain<Cell,Index>> Sync for MultiVoxelContainer<Index, Cell, Pos, Force, V, D> {}
