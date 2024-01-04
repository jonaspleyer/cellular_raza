use cellular_raza_concepts::cycle::CycleEvent;
use serde::{Deserialize, Serialize};

use std::{
    collections::VecDeque,
    ops::{Deref, DerefMut},
};

use super::CellIdentifier;

#[derive(Clone, Deserialize, Serialize)]
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
#[derive(Clone, Deserialize, Serialize)]
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
            next_random_mechanics_update: None,
        }
    }
}

impl<Pos, Vel, For, Float, const N: usize> UpdateMechanics<Pos, Vel, For, Float, N>
    for AuxStorageMechanics<Pos, Vel, For, Float, N>
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
    /// Set all cycle events. This function is currently the
    /// only way to change the contents of the stored events.
    fn set_cycle_events(&mut self, events: Vec<CycleEvent>);

    /// Get all cycle events currently stored.
    fn get_cycle_events(&self) -> Vec<CycleEvent>;

    /// Add another cycle event to the storage.
    fn add_cycle_event(&mut self, event: CycleEvent);
}

/// Stores intermediate information about the cell cycle.
#[derive(Clone, Deserialize, Serialize)]
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
    #[inline]
    fn set_cycle_events(&mut self, events: Vec<CycleEvent>) {
        self.cycle_events = events;
    }

    #[inline]
    fn get_cycle_events(&self) -> Vec<CycleEvent> {
        self.cycle_events.clone()
    }

    #[inline]
    fn add_cycle_event(&mut self, event: CycleEvent) {
        self.cycle_events.push(event);
    }
}

// --------------------------------- UPDATE-REACTIONS --------------------------------
/// Interface to store intermediate information about cellular reactions.
pub trait UpdateReactions<R> {
    fn set_conc(&mut self, conc: R);
    fn get_conc(&self) -> R;
    fn incr_conc(&mut self, incr: R);
}

#[derive(Clone, Deserialize, Serialize)]
pub struct AuxStorageReactions<R> {
    concentration: R,
}

impl<R> UpdateReactions<R> for AuxStorageReactions<R>
where
    R: Clone + core::ops::Add<R, Output = R>,
{
    #[inline]
    fn get_conc(&self) -> R {
        self.concentration.clone()
    }

    #[inline]
    fn incr_conc(&mut self, incr: R) {
        self.concentration = self.concentration.clone() + incr;
    }

    #[inline]
    fn set_conc(&mut self, conc: R) {
        self.concentration = conc;
    }
}

// -------------------------------- UPDATE-Interaction -------------------------------
/// Interface to store intermediate information about interactions.
pub trait UpdateInteraction {
    fn get_current_neighbours(&self) -> usize;
    fn set_current_neighbours(&mut self, neighbours: usize);
    fn incr_current_neighbours(&mut self, neighbours: usize);
}

#[derive(Clone, Deserialize, Serialize)]
pub struct AuxStorageInteraction {
    neighbour_count: usize,
}

impl UpdateInteraction for AuxStorageInteraction {
    fn get_current_neighbours(&self) -> usize {
        self.neighbour_count
    }

    fn incr_current_neighbours(&mut self, neighbours: usize) {
        self.neighbour_count += neighbours;
    }

    fn set_current_neighbours(&mut self, neighbours: usize) {
        self.neighbour_count = neighbours;
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
    fn mechanics_visibility_1() {}

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
    fn mechanics_visibility_2() {}

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
    fn mechanics_visibility_3() {}

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
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructMechanics<Pos, Vel, For, Float, const N: usize>
    /// where
    ///     Pos: Clone,
    /// {
    ///     #[UpdateMechanics(Pos, Vel, For, Float, N)]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, Float, N>,
    /// }
    /// ```
    fn mechanics_where_clause() {}

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

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// use cellular_raza_concepts::cycle::CycleEvent;
    ///
    /// #[derive(AuxStorage)]
    /// pub struct TestStructCycle {
    ///     #[UpdateCycle]
    ///     aux_cycle: AuxStorageCycle,
    /// }
    /// ```
    fn cycle_visibility_1() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// use cellular_raza_concepts::cycle::CycleEvent;
    ///
    /// #[derive(AuxStorage)]
    /// pub(crate) struct TestStructCycle {
    ///     #[UpdateCycle]
    ///     aux_cycle: AuxStorageCycle,
    /// }
    /// ```
    fn cycle_visibility_2() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// use cellular_raza_concepts::cycle::CycleEvent;
    ///
    /// #[derive(AuxStorage)]
    /// pub struct TestStructCycle<T> {
    ///     #[UpdateCycle]
    ///     aux_cycle: AuxStorageCycle,
    ///     _generic: T,
    /// }
    /// ```
    fn cycle_generic_param() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// use cellular_raza_concepts::cycle::CycleEvent;
    ///
    /// #[derive(AuxStorage)]
    /// pub struct TestStructCycle<const N: usize> {
    ///     #[UpdateCycle]
    ///     aux_cycle: AuxStorageCycle,
    ///     _generic: [f64; N],
    /// }
    /// ```
    fn cycle_const_generic_param() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// use cellular_raza_concepts::cycle::CycleEvent;
    ///
    /// #[derive(AuxStorage)]
    /// pub struct TestStructCycle<T>
    /// where
    ///     T: Clone,
    /// {
    ///     #[UpdateCycle]
    ///     aux_cycle: AuxStorageCycle,
    ///     _generic: T,
    /// }
    /// ```
    fn cycle_where_clause() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructReactions<R> {
    ///     #[UpdateReactions(R)]
    ///     aux_cycle: AuxStorageReactions<R>,
    /// }
    /// ```
    fn reactions_default() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// pub struct TestStructReactions<R> {
    ///     #[UpdateReactions(R)]
    ///     aux_cycle: AuxStorageReactions<R>,
    /// }
    /// ```
    fn reactions_visibility_1() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// pub(crate) struct TestStructReactions<R> {
    ///     #[UpdateReactions(R)]
    ///     aux_cycle: AuxStorageReactions<R>,
    /// }
    /// ```
    fn reactions_visibility_2() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructReactions<T, R> {
    ///     #[UpdateReactions(R)]
    ///     aux_cycle: AuxStorageReactions<R>,
    ///     generic: T,
    /// }
    /// ```
    fn reactions_generic_param() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructReactions<R, const N: usize> {
    ///     #[UpdateReactions(R)]
    ///     aux_cycle: AuxStorageReactions<R>,
    ///     generic_array: [usize; N],
    /// }
    /// ```
    fn reactions_const_generic_param() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructReactions<T, R>
    /// where
    ///     T: Clone,
    /// {
    ///     #[UpdateReactions(R)]
    ///     aux_cycle: AuxStorageReactions<R>,
    ///     generic: T,
    /// }
    /// ```
    fn reactions_where_clause() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructInteraction {
    ///     #[UpdateInteraction]
    ///     aux_interaction: AuxStorageInteraction,
    /// }
    /// ```
    fn interactions_default() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// pub struct TestStructInteraction {
    ///     #[UpdateInteraction]
    ///     aux_interaction: AuxStorageInteraction,
    /// }
    /// ```
    fn interactions_visibility_1() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// pub(crate) struct TestStructInteraction {
    ///     #[UpdateInteraction]
    ///     aux_interaction: AuxStorageInteraction,
    /// }
    /// ```
    fn interactions_visibility_2() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructInteraction<T> {
    ///     #[UpdateInteraction]
    ///     aux_interaction: AuxStorageInteraction,
    ///     generic: T,
    /// }
    /// ```
    fn interactions_generic_param() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructInteraction<const N: usize> {
    ///     #[UpdateInteraction]
    ///     aux_interaction: AuxStorageInteraction,
    ///     generic: [f64; N],
    /// }
    /// ```
    fn interactions_const_generic_param() {}

    /// ```
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructInteraction<T>
    /// where
    ///     T: Clone,
    /// {
    ///     #[UpdateInteraction]
    ///     aux_interaction: AuxStorageInteraction,
    ///     generic: T,
    /// }
    /// ```
    fn interactions_where_clause() {}
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
#[derive(Clone, Deserialize, Serialize)]
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

#[cfg(feature = "never")]
mod future_ring_buffer {
    // Continue working in this playground:
    // https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=e4f1b291da1fb3f6a303ec5c55930319

    #[derive(Debug)]
    struct FixedSizeRingBuffer<T, const N: usize> {
        items: [std::mem::MaybeUninit<T>; N],
        first: usize,
        size: usize,
    }

    impl<T, const N: usize> Clone for FixedSizeRingBuffer<T, N> {
        fn clone(&self) -> Self {
            Self {
                items: self.items.clone(),
                first: self.first,
                size: self.size,
            }
        }
    }

    impl<T, const N: usize> Serialize for FixedSizeRingBuffer<T, N>
    where
        T: Serialize,
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            use serde::ser::SerializeSeq;
            let mut s = serializer.serialize_seq(Some(self.size))?;
            for element in self.iter() {
                s.serialize_element(element)?;
            }
            s.end()
        }
    }

    impl<'de, T, const N: usize> Deserialize<'de> for FixedSizeRingBuffer<T, N>
    where
        T: for<'a> Deserialize<'de>,
    {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            todo!()
        }
    }

    struct FixedSizeRingBufferIter<'a, T, const N: usize> {
        items: &'a [std::mem::MaybeUninit<T>; N],
        current: usize,
        left_size: usize,
    }

    impl<'a, T, const N: usize> Iterator for FixedSizeRingBufferIter<'a, T, N> {
        type Item = &'a T;

        fn next(&mut self) -> Option<&'a T> {
            if self.left_size == 0 {
                return None;
            }
            let index = self.current;
            self.current = (self.current + 1) % N;
            self.left_size -= 1;
            Some(unsafe { self.items[index].assume_init_ref() })
        }
    }

    impl<T, const N: usize> FixedSizeRingBuffer<T, N> {
        fn push(&mut self, new_item: T) {
            self.items[self.first].write(new_item);
            self.first = (self.first + 1) % N;
            self.size = N.min(self.size + 1);
        }

        fn iter<'a>(&'a self) -> FixedSizeRingBufferIter<'a, T, N> {
            FixedSizeRingBufferIter {
                items: &self.items,
                current: self.first,
                left_size: self.size,
            }
        }
    }

    impl<T, const N: usize> FixedSizeRingBuffer<T, N> {
        fn new() -> Self {
            Self {
                items: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
                first: 0,
                size: 0,
            }
        }
    }
}
