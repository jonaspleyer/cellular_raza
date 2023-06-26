use cellular_raza_concepts::cycle::CycleEvent;
use serde::{Deserialize, Serialize};


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
