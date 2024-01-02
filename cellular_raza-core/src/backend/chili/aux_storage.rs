use cellular_raza_concepts::cycle::CycleEvent;
use serde::{Deserialize, Serialize};

use std::{
    collections::VecDeque,
    ops::{Deref, DerefMut},
};

use super::CellIdentifier;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CellBox<C> {
    /// The identifier is composed of two values, one for the voxel index in which the
    /// object was created and another one which counts how many elements have already
    /// been created there.
    pub identifier: CellIdentifier,
    /// Identifier of the parent cell if this cell was created by cell-division
    pub parent: Option<CellIdentifier>,
    /// The cell which is encapsulated by this box.
    pub cell: C,
}

impl<C> Deref for CellBox<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.cell
    }
}

impl<C> DerefMut for CellBox<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cell
    }
}

impl<C> cellular_raza_concepts::domain_new::Id for CellBox<C> {
    type Identifier = CellIdentifier;

    fn get_id(&self) -> CellIdentifier {
        self.identifier
    }
}

impl<C> CellBox<C> {
    pub(crate) fn new(identifier: CellIdentifier, parent: Option<CellIdentifier>, cell: C) -> Self {
        Self {
            identifier,
            parent,
            cell,
        }
    }
}

// --------------------------------- UPDATE-MECHANICS --------------------------------
/// Used to store intermediate information about last positions and velocities.
/// Can store up to `N` values.
pub trait UpdateMechanics<Pos, Vel, For, Float, const N: usize> {
    /// Stores the last position of the cell. May overwrite old results depending on
    /// how many old results are being stored.
    fn set_last_position(&mut self, pos: Pos);

    /// Get all previous positions. This number maybe smaller than the maximum number of stored
    /// positions but never exceeds it.
    fn previous_positions(&self) -> std::collections::vec_deque::Iter<Pos>;

    /// Stores the last velocity of the cell. Overwrites old results when stored amount
    /// exceeds number of maximum stored values.
    fn set_last_velocity(&mut self, vel: Vel);

    /// Get all previous velocities. This number may be smaller than the maximum number of stored
    /// velocities but never exceeds it.
    fn previous_velocities(&self) -> std::collections::vec_deque::Iter<Vel>;

    /// Add force to currently stored forces
    fn add_force(&mut self, force: For);

    /// Obtain current force on cell
    fn get_current_force(&self) -> For;

    /// Removes all stored forces
    fn clear_forces(&mut self);

    /// Next time point at which the internal state is updated randomly
    fn get_next_random_update(&self) -> Option<Float>;

    /// Set the time point for the next random update
    fn set_next_random_update(&mut self, next: Option<Float>);
}

/// Stores intermediate information about the mechanics of a cell.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AuxStorageMechanics<Pos, Vel, For, Float, const N: usize> {
    positions: FixedSizeRingBuffer<Pos, N>,
    velocities: FixedSizeRingBuffer<Vel, N>,
    current_force: For,
    next_random_mechanics_update: Option<Float>,
}

impl<Pos, Vel, For, Float, const N: usize> Default for AuxStorageMechanics<Pos, Vel, For, Float, N>
where
    For: num::Zero,
{
    fn default() -> Self {
        AuxStorageMechanics {
            positions: FixedSizeRingBuffer::new(),
            velocities: FixedSizeRingBuffer::new(),
            current_force: For::zero(),
            next_random_mechanics_update: None
        }
    }
}

impl<Pos, Vel, For, Float, const N: usize> UpdateMechanics<Pos, Vel, For, Float, N> for AuxStorageMechanics<Pos, Vel, For, Float, N>
where
    For: Clone + core::ops::AddAssign<For> + num::Zero,
    Float: Clone,
{
    #[inline]
    fn previous_positions(&self) -> std::collections::vec_deque::Iter<Pos> {
        self.positions.iter()
    }

    #[inline]
    fn previous_velocities(&self) -> std::collections::vec_deque::Iter<Vel> {
        self.velocities.iter()
    }

    #[inline]
    fn set_last_position(&mut self, pos: Pos) {
        self.positions.push(pos);
    }

    #[inline]
    fn set_last_velocity(&mut self, vel: Vel) {
        self.velocities.push(vel);
    }

    #[inline]
    fn add_force(&mut self, force: For) {
        self.current_force += force;
    }

    #[inline]
    fn get_current_force(&self) -> For {
        self.current_force.clone()
    }

    #[inline]
    fn clear_forces(&mut self) {
        self.current_force = For::zero();
    }

    #[inline]
    fn get_next_random_update(&self) -> Option<Float> {
        self.next_random_mechanics_update.clone()
    }

    #[inline]
    fn set_next_random_update(&mut self, next: Option<Float>) {
        self.next_random_mechanics_update = next;
    }
}

// ----------------------------------- UPDATE-CYCLE ----------------------------------
/// Trait which describes how to store intermediate
/// information on the cell cycle.
pub trait UpdateCycle {
    /// Set all cycle events. This function is currently the only way to change the contents of the stored events.
    fn set_cycle_events(&mut self, events: Vec<CycleEvent>);

    /// Get all cycle events currently stored.
    fn get_cycle_events(&self) -> Vec<CycleEvent>;

    /// Add another cycle event to the storage.
    fn add_cycle_event(&mut self, event: CycleEvent);
}

/// Stores intermediate information about the cell cycle.
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

#[allow(unused)]
#[doc(hidden)]
mod test_derive_compile {
    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructMechanics<Pos, Vel, For, Float, const N: usize> {
    ///     #[UpdateMechanics(Pos, Vel, For, Float, N)]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, Float, N>,
    /// }
    /// ```
    fn mechanics_default() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// pub struct TestStructMechanics<Pos, Vel, For, Float, const N: usize> {
    ///     #[UpdateMechanics(Pos, Vel, For, Float, N)]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, Float, N>,
    /// }
    /// ```
    fn mechanics_vis_1() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// pub(crate) struct TestStructMechanics<Pos, Vel, For, Float, const N: usize> {
    ///     #[UpdateMechanics(Pos, Vel, For, Float, N)]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, Float, N>,
    /// }
    /// ```
    fn mechanics_vis_2() {}

    /// ```
    /// mod some_module {
    ///     use cellular_raza_core_derive::*;
    ///     use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    ///     #[derive(AuxStorage)]
    ///     pub(super) struct TestStructMechanics<Pos, Vel, For, Float, const N: usize> {
    ///         #[UpdateMechanics(Pos, Vel, For, Float, N)]
    ///         aux_mechanics: AuxStorageMechanics<Pos, Vel, For, Float, N>,
    ///     }
    /// }
    /// fn use_impl<T, Pos, Vel, For, Float, const N: usize>(
    ///     aux_storage: T
    /// ) -> For
    /// where
    ///     T: cellular_raza_core::backend::chili::aux_storage::UpdateMechanics<Pos, Vel, For, Float, N>,
    /// {
    ///     aux_storage.get_current_force()
    /// }
    /// ```
    fn mechanics_vis_3() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructMechanics<Pos, Vel, For, Float, T, const N: usize> {
    ///     #[UpdateMechanics(Pos, Vel, For, Float, N)]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, Float, N>,
    ///     other: T,
    /// }
    /// ```
    fn mechanics_more_struct_generics() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructMechanics<Pos, Vel, For, Float, const N: usize, const M: usize> {
    ///     #[UpdateMechanics(Pos, Vel, For, Float, N)]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, Float, N>,
    ///     count: [i64; M],
    /// }
    /// ```
    fn mechanics_more_struct_const_generics() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// use cellular_raza_concepts::cycle::CycleEvent;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructCycle {
    ///     #[UpdateCycle]
    ///     aux_cycle: AuxStorageCycle,
    /// }
    /// ```
    fn cycle_default() {}
}

#[cfg(test)]
pub mod test_derive_aux_storage {
    use super::*;
    use cellular_raza_core_derive::AuxStorage;

    #[derive(AuxStorage)]
    struct TestStructDouble<Pos, Vel, For, Float, const N: usize> {
        #[UpdateCycle]
        aux_cycle: AuxStorageCycle,
        #[UpdateMechanics(Pos, Vel, For, Float, N)]
        aux_mechanics: AuxStorageMechanics<Pos, Vel, For, Float, N>,
    }

    #[derive(AuxStorage)]
    struct TestStructCycle {
        #[UpdateCycle]
        aux_cycle: AuxStorageCycle,
    }

    #[derive(AuxStorage)]
    struct TestStructMechanics<Pos, Vel, For, Float, const N: usize> {
        #[UpdateMechanics(Pos, Vel, For, Float, N)]
        aux_mechanis: AuxStorageMechanics<Pos, Vel, For, Float, N>,
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
        let mut aux_storage = TestStructMechanics::<_, _, _, _, 4> {
            aux_mechanis: AuxStorageMechanics::default(),
        };
        aux_storage.set_last_position(3_f64);
        aux_storage.set_last_velocity(5_f32);
        aux_storage.add_force(1_f32);
        aux_storage.set_next_random_update(Some(120.034958_f32));
    }

    #[test]
    fn cycle_mechanics_add_get_events() {
        let mut aux_storage = TestStructDouble::<_, _, _, _, 4> {
            aux_cycle: AuxStorageCycle::default(),
            aux_mechanics: AuxStorageMechanics::default(),
        };
        aux_storage.set_last_position(3_f64);
        aux_storage.set_last_velocity(5_f32);
        aux_storage.add_force(-5_f32);
        aux_storage.set_next_random_update(Some(33_f32));
        add_get_events(&mut aux_storage);
    }

    #[test]
    fn cycle_mechanics_set_get_events() {
        let mut aux_storage = TestStructDouble::<_, _, _, _, 4> {
            aux_cycle: AuxStorageCycle::default(),
            aux_mechanics: AuxStorageMechanics::default(),
        };
        aux_storage.set_last_position(3_f64);
        aux_storage.set_last_velocity(5_f32);
        aux_storage.add_force(111_i64);
        aux_storage.set_next_random_update(Some(0_f32));
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
    /// Constructs a new empty ringbuffer.
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

    /// Iterate over elements stored in the [FixedSizeRingBuffer].
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
