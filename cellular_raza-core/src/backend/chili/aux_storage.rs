use cellular_raza_concepts::{
    cycle::CycleEvent,
    errors::{BoundaryError, CalcError},
};
use serde::{Deserialize, Serialize};

use std::{collections::VecDeque, hash::Hash};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::concepts::SubDomain;

pub struct SubDomainBox<S, C, A>
where
    S: SubDomain<C>,
{
    subdomain: S,
    voxels: std::collections::BTreeMap<S::VoxelIndex, Voxel<C, A>>,
}

impl<S, C, A> SubDomainBox<S, C, A>
where
    S: SubDomain<C>,
{
    pub fn from_subdomain_and_cells(subdomain: S, cells: Vec<C>) -> Self
    where
        S::VoxelIndex: std::cmp::Eq + Hash + Ord,
        A: Default,
    {
        let voxel_indices = subdomain.get_all_indices();
        // TODO let voxels = subdomain.generate_all_voxels();
        let mut index_to_cells = cells
            .into_iter()
            .map(|cell| (subdomain.get_voxel_index_of(&cell).unwrap(), cell))
            .fold(
                std::collections::HashMap::new(),
                |mut acc, (index, cell)| {
                    let cells_in_voxel = acc.entry(index).or_insert(Vec::new());
                    cells_in_voxel.push((cell, A::default()));
                    acc
                },
            );
        let voxels = voxel_indices
            .into_iter()
            .map(|index| {
                let rng = ChaCha8Rng::seed_from_u64(1);
                let cells = index_to_cells.remove(&index).or(Some(Vec::new())).unwrap();
                (
                    index,
                    Voxel {
                        cells,
                        new_cells: Vec::new(),
                        id_counter: 0,
                        rng,
                    },
                )
            })
            .collect();
        Self { subdomain, voxels }
    }

    pub fn apply_boundary(&mut self) -> Result<(), BoundaryError> {
        self.voxels
            .iter_mut()
            .map(|(_, voxel)| voxel.cells.iter_mut())
            .flatten()
            .map(|(cell, _)| self.subdomain.apply_boundary(cell))
            .collect::<Result<(), BoundaryError>>()
    }

    // TODO this is not a boundary error!
    pub fn insert_cells(&mut self, new_cells: &mut Vec<(C, A)>) -> Result<(), BoundaryError>
    where
        S::VoxelIndex: Ord,
    {
        for cell in new_cells.drain(..) {
            let voxel_index = self.subdomain.get_voxel_index_of(&cell.0)?;
            self.voxels
                .get_mut(&voxel_index)
                .ok_or(BoundaryError {
                    message: "Could not find correct voxel for cell".to_owned(),
                })?
                .cells
                .push(cell);
        }
        Ok(())
    }

    pub fn update_cycle(&mut self) -> Result<(), CalcError>
    where
        C: cellular_raza_concepts::cycle::Cycle<C>,
        A: UpdateCycle,
    {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        self.voxels
            .iter_mut()
            .map(|(_, voxel)| voxel.cells.iter_mut())
            .flatten()
            .for_each(|(cell, aux_storage)| {
                if let Some(event) = C::update_cycle(&mut rng, &0.01, cell) {
                    aux_storage.add_cycle_event(event);
                }
            });
        Ok(())
    }
}

pub trait UpdateMechanics<P, V, const N: usize> {
    fn set_last_position(&mut self, pos: P);
    fn previous_positions(&self) -> std::collections::vec_deque::Iter<P>;
    /* fn last_position(&self) -> Option<P>
    where
        P: Clone,
    {
        self.previous_positions().iter().last().and_then(|x| Some(x.clone()))
    }*/
    fn set_last_velocity(&mut self, vel: V);
    fn previous_velocities(&self) -> std::collections::vec_deque::Iter<V>;
    /* fn last_velocity(&self) -> Option<V>
    where
        V: Clone,
    {
        self.previous_velocities().iter().last().and_then(|x| Some(x.clone()))
    }*/
}

pub trait UpdateCycle {
    fn set_cycle_events(&mut self, events: Vec<CycleEvent>);
    fn get_cycle_events(&self) -> Vec<CycleEvent>;
    fn add_cycle_event(&mut self, event: CycleEvent);
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AuxStorage<T>(T);

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AuxStorageCycle {
    cycle_events: Vec<CycleEvent>,
}

impl Default for AuxStorageCycle {
    fn default() -> Self {
        AuxStorageCycle {
            cycle_events: Vec::new(),
        }
    }
}

impl UpdateCycle for AuxStorageCycle {
    fn set_cycle_events(&mut self, events: Vec<CycleEvent>) {
        self.cycle_events = events;
    }

    fn get_cycle_events(&self) -> Vec<CycleEvent> {
        self.cycle_events.clone()
    }

    fn add_cycle_event(&mut self, event: CycleEvent) {
        self.cycle_events.push(event);
    }
}

impl<T> UpdateCycle for AuxStorage<T>
where
    T: UpdateCycle,
{
    fn add_cycle_event(&mut self, event: CycleEvent) {
        self.0.add_cycle_event(event)
    }

    fn set_cycle_events(&mut self, events: Vec<CycleEvent>) {
        self.0.set_cycle_events(events)
    }

    fn get_cycle_events(&self) -> Vec<CycleEvent> {
        self.0.get_cycle_events()
    }
}

impl<T, U> UpdateCycle for (T, U)
where
    U: UpdateCycle,
{
    fn set_cycle_events(&mut self, events: Vec<CycleEvent>) {
        self.1.set_cycle_events(events)
    }

    fn get_cycle_events(&self) -> Vec<CycleEvent> {
        self.1.get_cycle_events()
    }

    fn add_cycle_event(&mut self, event: CycleEvent) {
        self.1.add_cycle_event(event)
    }
}

#[cfg(test)]
mod test_aux_storage_cycle {
    use super::*;

    #[test]
    fn nested_1() {
        let aux_storage = AuxStorage(AuxStorageCycle {cycle_events: Vec::new()});

        let events = aux_storage.get_cycle_events();
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn nested_2() {
        let aux_storage = AuxStorage((
            AuxStorage(1_f64),
            AuxStorageCycle {
                cycle_events: Vec::new(),
            },
        ));

        let events = aux_storage.get_cycle_events();
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn nested_3() {
        let aux_storage = AuxStorage((
            AuxStorage(1_f64),
            AuxStorage((
                1_f64,
                AuxStorageCycle {
                    cycle_events: Vec::new(),
                },
            )),
        ));

        let events = aux_storage.get_cycle_events();
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn nested_4() {
        let aux_storage = AuxStorage((
            AuxStorage(1_f64),
            AuxStorage((
                1_f64,
                AuxStorage((
                    "This is my home".to_owned(),
                    AuxStorageCycle {
                        cycle_events: Vec::new(),
                    },
                ))
            )),
        ));

        let events = aux_storage.get_cycle_events();
        assert_eq!(events.len(), 0);
    }
}

/// Small implementation of a ring Buffer with constant size.
/// Makes use of a fixed-size array internally.
/// ```
/// # use cellular_raza_core::backend::chili::aux_storage::FixedSizeRingBuffer;
/// let mut ring_buffer = FixedSizeRingBuffer::<i64, 4>::new();
///
/// // These entries will be made into free space in the buffer.
/// ring_buffer.push(1_i64);
/// ring_buffer.push(2_i64);
/// ring_buffer.push(3_i64);
/// ring_buffer.push(4_i64);
///
/// // Now it will start truncating initial entries.
/// ring_buffer.push(5_i64);
/// ring_buffer.push(6_i64);
/// ring_buffer.push(7_i64);
///
/// let mut elements = ring_buffer.iter();
/// assert_eq!(elements.next(), Some(&4));
/// assert_eq!(elements.next(), Some(&5));
/// assert_eq!(elements.next(), Some(&6));
/// assert_eq!(elements.next(), Some(&7));
///
/// ring_buffer.push(8);
/// let _ = ring_buffer.into_iter();
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FixedSizeRingBuffer<T, const N: usize> {
    /// contains the elements in the buffer
    values: VecDeque<T>,
}

impl<T, const N: usize> FixedSizeRingBuffer<T, N> {
    pub fn new() -> Self {
        Self {
            values: VecDeque::with_capacity(N),
        }
    }

    /// Push new entries to the back while truncating if needed.
    pub fn push(&mut self, value: T) {
        if self.values.len() >= N {
            self.values.pop_front();
        }
        self.values.push_back(value);
    }

    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_, T> {
        self.into_iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a FixedSizeRingBuffer<T, N> {
    type Item = &'a T;
    type IntoIter = std::collections::vec_deque::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<T, const N: usize> IntoIterator for FixedSizeRingBuffer<T, N> {
    type Item = T;
    type IntoIter = std::collections::vec_deque::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AuxStorageMechanics<P, V, const N: usize> {
    positions: FixedSizeRingBuffer<P, N>,
    velocities: FixedSizeRingBuffer<V, N>,
}

impl<P, V, const N: usize> Default for AuxStorageMechanics<P, V, N> {
    fn default() -> Self {
        AuxStorageMechanics {
            positions: FixedSizeRingBuffer::new(),
            velocities: FixedSizeRingBuffer::new(),
        }
    }
}

impl<P, V, const N: usize> UpdateMechanics<P, V, N> for AuxStorageMechanics<P, V, N> {
    fn previous_positions(&self) -> std::collections::vec_deque::Iter<P> {
        self.positions.iter()
    }

    fn previous_velocities(&self) -> std::collections::vec_deque::Iter<V> {
        self.velocities.iter()
    }

    fn set_last_position(&mut self, pos: P) {
        self.positions.push(pos);
    }

    fn set_last_velocity(&mut self, vel: V) {
        self.velocities.push(vel);
    }
}

impl<T, P, V, const N: usize> UpdateMechanics<P, V, N> for AuxStorage<T>
where
    T: UpdateMechanics<P, V, N>,
{
    fn previous_positions(&self) -> std::collections::vec_deque::Iter<P> {
        self.0.previous_positions()
    }

    fn previous_velocities(&self) -> std::collections::vec_deque::Iter<V> {
        self.0.previous_velocities()
    }

    fn set_last_position(&mut self, pos: P) {
        self.0.set_last_position(pos);
    }

    fn set_last_velocity(&mut self, vel: V) {
        self.0.set_last_velocity(vel);
    }
}

impl<T, U, P, V, const N: usize> UpdateMechanics<P, V, N> for (T, U)
where
    U: UpdateMechanics<P, V, N>,
{
    fn previous_positions(&self) -> std::collections::vec_deque::Iter<P> {
        self.1.previous_positions()
    }

    fn previous_velocities(&self) -> std::collections::vec_deque::Iter<V> {
        self.1.previous_velocities()
    }

    fn set_last_position(&mut self, pos: P) {
        self.1.set_last_position(pos);
    }

    fn set_last_velocity(&mut self, vel: V) {
        self.1.set_last_velocity(vel);
    }
}

#[cfg(test)]
pub mod test_aux_storage_mechanics {
    use super::*;

    #[test]
    fn nested_1() {
        let mut aux_storage = AuxStorage(
            AuxStorageMechanics::<_, _, 4>::default(),
        );

        aux_storage.set_last_position(1_f64);
        aux_storage.set_last_velocity(0.1_f32);

        let mut positions = aux_storage.previous_positions();
        let mut velocities = aux_storage.previous_velocities();

        assert_eq!(positions.next(), Some(&1_f64));
        assert_eq!(velocities.next(), Some(&0.1_f32));
    }

    #[test]
    fn nested_2() {
        let mut aux_storage = AuxStorage((
            AuxStorage(1_f64),
            AuxStorageMechanics::<_, _, 4>::default(),
        ));

        aux_storage.set_last_position(1_f64);
        aux_storage.set_last_velocity(0.1_f32);

        let mut positions = aux_storage.previous_positions();
        let mut velocities = aux_storage.previous_velocities();

        assert_eq!(positions.next(), Some(&1_f64));
        assert_eq!(velocities.next(), Some(&0.1_f32));
    }

    #[test]
    fn nested_3() {
        let mut aux_storage = AuxStorage((
            AuxStorage(1_f64),
            AuxStorage((
                "All your base belong to us".to_owned(),
                AuxStorageMechanics::<_, _, 4>::default()
            )),
        ));

        aux_storage.set_last_position(1_f64);
        aux_storage.set_last_velocity(0.1_f32);

        let mut positions = aux_storage.previous_positions();
        let mut velocities = aux_storage.previous_velocities();

        assert_eq!(positions.next(), Some(&1_f64));
        assert_eq!(velocities.next(), Some(&0.1_f32));
    }

    #[test]
    fn nested_4() {
        let mut aux_storage = AuxStorage((
            AuxStorage(1_f64),
            AuxStorage((
                "All your base belong to us".to_owned(),
                AuxStorage((
                    1_f64,
                    AuxStorageMechanics::<_, _, 4>::default()
                )),
            )),
        ));

        aux_storage.set_last_position(1_f64);
        aux_storage.set_last_velocity(0.1_f32);

        let mut positions = aux_storage.previous_positions();
        let mut velocities = aux_storage.previous_velocities();

        assert_eq!(positions.next(), Some(&1_f64));
        assert_eq!(velocities.next(), Some(&0.1_f32));
    }
}

pub struct Voxel<C, A> {
    pub cells: Vec<(C, A)>,
    pub new_cells: Vec<C>,
    pub id_counter: u64,
    pub rng: rand_chacha::ChaCha8Rng,
}
