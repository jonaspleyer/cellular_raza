use cellular_raza_concepts::{
    cycle::CycleEvent,
    errors::{BoundaryError, CalcError},
};
use serde::{Deserialize, Serialize};

use std::{collections::VecDeque, hash::Hash};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::concepts::SubDomain;
use super::simulation_flow::{BarrierSync, SyncSubDomains};

pub struct SubDomainBox<S, C, A, Sy = BarrierSync>
where
    S: SubDomain<C>,
{
    pub(crate) subdomain: S,
    pub(crate) voxels: std::collections::BTreeMap<S::VoxelIndex, Voxel<C, A>>,
    pub(crate) syncer: Sy,
}

impl<S, C, A, Sy> SubDomainBox<S, C, A, Sy>
where
    S: SubDomain<C>,
{
    pub fn initialize(subdomain: S, cells: Vec<C>, syncer: Sy, rng_seed: u64) -> Self
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
                let rng = ChaCha8Rng::seed_from_u64(rng_seed);
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
        Self {
            subdomain,
            voxels,
            syncer,
        }
    }

    pub fn sync(&mut self)
    where
        Sy: SyncSubDomains,
    {
        self.syncer.sync();
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
    pub fn insert_cells(&mut self, new_cells: &mut Vec<(C, Option<A>)>) -> Result<(), BoundaryError>
    where
        S::VoxelIndex: Ord,
        A: Default,
    {
        for cell in new_cells.drain(..) {
            let voxel_index = self.subdomain.get_voxel_index_of(&cell.0)?;
            self.voxels
                .get_mut(&voxel_index)
                .ok_or(BoundaryError {
                    message: "Could not find correct voxel for cell".to_owned(),
                })?
                .cells
                .push((cell.0, cell.1.map_or(A::default(), |x| x)));
        }
        Ok(())
    }

    pub fn update_cycle(&mut self) -> Result<(), CalcError>
    where
        C: cellular_raza_concepts::cycle::Cycle<C>,
        A: UpdateCycle,
    {
        self.voxels.iter_mut().for_each(|(_, voxel)| {
            voxel.cells.iter_mut().for_each(|(cell, aux_storage)| {
                if let Some(event) = C::update_cycle(&mut voxel.rng, &0.01, cell) {
                    aux_storage.add_cycle_event(event);
                }
            })
        });
        Ok(())
    }
}

/// Used to store intermediate information about last positions and velocities.
/// Can store up to `N` values.
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

#[cfg(test)]
pub mod test_derive_aux_storage {
    use super::*;
    use cellular_raza_core_derive::AuxStorage;

    #[derive(AuxStorage)]
    struct TestStructDouble<P, V> {
        #[UpdateCycle]
        aux_cycle: AuxStorageCycle,
        #[UpdateMechanics]
        aux_mechanics: AuxStorageMechanics<P, V, 4>,
    }

    #[derive(AuxStorage)]
    struct TestStructCycle {
        #[UpdateCycle]
        aux_cycle: AuxStorageCycle,
    }

    #[derive(AuxStorage)]
    struct TestStructMechanics<P, V> {
        #[UpdateMechanics]
        aux_mechanis: AuxStorageMechanics<P, V, 4>,
    }

    fn add_get_events<A>(aux_storage: &mut A)
    where
        A: UpdateCycle,
    {
        aux_storage.add_cycle_event(CycleEvent::Division);
        let events = aux_storage.get_cycle_events();
        assert_eq!(events, vec![CycleEvent::Division]);
    }

    fn set_get_events<A>(aux_storage: &mut A)
    where
        A: UpdateCycle,
    {
        let initial_events = vec![
            CycleEvent::Division,
            CycleEvent::Division,
            CycleEvent::PhasedDeath,
        ];
        aux_storage.set_cycle_events(initial_events.clone());
        let events = aux_storage.get_cycle_events();
        assert_eq!(events.len(), 3);
        assert_eq!(events, initial_events);
    }

    #[test]
    fn cycle_add_get_events() {
        let mut aux_storage = TestStructCycle {
            aux_cycle: AuxStorageCycle::default(),
        };
        add_get_events(&mut aux_storage);
    }

    #[test]
    fn cycle_set_get_events() {
        let mut aux_storage = TestStructCycle {
            aux_cycle: AuxStorageCycle::default(),
        };
        set_get_events(&mut aux_storage);
    }

    #[test]
    fn mechanics() {
        let mut aux_storage = TestStructMechanics {
            aux_mechanis: AuxStorageMechanics::default(),
        };
        aux_storage.set_last_position(3_f64);
        aux_storage.set_last_velocity(5_f32);
    }

    #[test]
    fn cycle_mechanics_add_get_events() {
        let mut aux_storage = TestStructDouble::<_, _> {
            aux_cycle: AuxStorageCycle::default(),
            aux_mechanics: AuxStorageMechanics::default(),
        };
        aux_storage.set_last_position(3_f64);
        aux_storage.set_last_velocity(5_f32);
        add_get_events(&mut aux_storage);
    }

    #[test]
    fn cycle_mechanics_set_get_events() {
        let mut aux_storage = TestStructDouble::<_, _> {
            aux_cycle: AuxStorageCycle::default(),
            aux_mechanics: AuxStorageMechanics::default(),
        };
        aux_storage.set_last_position(3_f64);
        aux_storage.set_last_velocity(5_f32);
        set_get_events(&mut aux_storage);
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
    // TODO can we do this without heap allocations?
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

pub struct Voxel<C, A> {
    pub cells: Vec<(C, A)>,
    pub new_cells: Vec<C>,
    pub id_counter: u64,
    pub rng: rand_chacha::ChaCha8Rng,
}
