use cellular_raza_concepts::cycle::CycleEvent;
use serde::{Deserialize, Serialize};

use std::ops::{Deref, DerefMut};

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
    fn previous_positions<'a>(&'a self) -> FixedSizeRingBufferIter<'a, Pos, N>;

    /// Stores the last velocity of the cell. Overwrites old results when stored amount
    /// exceeds number of maximum stored values.
    fn set_last_velocity(&mut self, vel: Vel);

    /// Get all previous velocities. This number may be smaller than the maximum number of stored
    /// velocities but never exceeds it.
    fn previous_velocities<'a>(&'a self) -> FixedSizeRingBufferIter<'a, Vel, N>;

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
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct AuxStorageMechanics<Pos, Vel, For, Float, const N: usize> {
    positions: FixedSizeRingBuffer<Pos, N>,
    velocities: FixedSizeRingBuffer<Vel, N>,
    current_force: For,
    next_random_mechanics_update: Option<Float>,
}

impl<Pos, Vel, For, Float, const N: usize> UpdateMechanics<Pos, Vel, For, Float, N>
    for AuxStorageMechanics<Pos, Vel, For, Float, N>
where
    For: Clone + core::ops::AddAssign<For> + num::Zero,
    Float: Clone,
{
    #[inline]
    fn previous_positions<'a>(&'a self) -> FixedSizeRingBufferIter<'a, Pos, N> {
        self.positions.iter()
    }

    #[inline]
    fn previous_velocities<'a>(&'a self) -> FixedSizeRingBufferIter<'a, Vel, N> {
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
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct AuxStorageCycle {
    cycle_events: Vec<CycleEvent>,
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

#[derive(Clone, Default, Deserialize, Serialize)]
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

#[derive(Clone, Default, Deserialize, Serialize)]
pub struct AuxStorageInteraction {
    neighbour_count: usize,
}

impl UpdateInteraction for AuxStorageInteraction {
    #[inline]
    fn get_current_neighbours(&self) -> usize {
        self.neighbour_count
    }

    #[inline]
    fn incr_current_neighbours(&mut self, neighbours: usize) {
        self.neighbour_count += neighbours;
    }

    #[inline]
    fn set_current_neighbours(&mut self, neighbours: usize) {
        self.neighbour_count = neighbours;
    }
}

#[allow(unused)]
#[doc(hidden)]
mod test_derive_aux_storage_compile {
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
/// let mut ring_buffer = FixedSizeRingBuffer::<i64, 4>::default();
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
/// ```
#[derive(Debug)]
pub struct FixedSizeRingBuffer<T, const N: usize> {
    items: [std::mem::MaybeUninit<T>; N],
    first: usize,
    size: usize,
}

impl<T, const N: usize> Clone for FixedSizeRingBuffer<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        let mut new_items: [std::mem::MaybeUninit<T>; N] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..self.size {
            let i = (self.first + i) % N;
            new_items[i].write(unsafe { self.items[i].assume_init_ref().clone() });
        }

        Self {
            items: new_items,
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

struct FixedSizedRingBufferVisitor<T, const N: usize> {
    phantom: core::marker::PhantomData<T>,
}

impl<'de, T, const N: usize> serde::de::Visitor<'de> for FixedSizedRingBufferVisitor<T, N>
where
    T: Deserialize<'de>,
{
    type Value = Vec<T>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str(&format!(
            "{} or less values of the type {}",
            N,
            std::any::type_name::<T>()
        ))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut elements = Vec::new();
        while let Some(element) = seq.next_element()? {
            elements.push(element);
        }
        Ok(elements)
    }
}

impl<'de, T, const N: usize> Deserialize<'de> for FixedSizeRingBuffer<T, N>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let elements = deserializer.deserialize_seq(FixedSizedRingBufferVisitor::<T, N> {
            phantom: core::marker::PhantomData,
        })?;
        let mut ring_buffer = FixedSizeRingBuffer::default();
        if elements.len() > N {
            todo!()
        }
        for element in elements.into_iter() {
            ring_buffer.push(element);
        }
        Ok(ring_buffer)
    }
}

pub struct FixedSizeRingBufferIter<'a, T, const N: usize> {
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

impl<T, const N: usize> Default for FixedSizeRingBuffer<T, N> {
    fn default() -> Self {
        Self {
            items: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            first: 0,
            size: 0,
        }
    }
}

impl<T, const N: usize> FixedSizeRingBuffer<T, N> {
    pub fn push(&mut self, new_item: T) {
        let last = (self.first + self.size) % N;
        self.items[last].write(new_item);
        self.first = (self.first + self.size.div_euclid(N)) % N;
        self.size = N.min(self.size + 1);
    }

    pub fn iter<'a>(&'a self) -> FixedSizeRingBufferIter<'a, T, N> {
        FixedSizeRingBufferIter {
            items: &self.items,
            current: self.first,
            left_size: self.size,
        }
    }
}

#[cfg(test)]
mod test_ring_buffer {
    use super::*;

    #[test]
    fn test_pushing_full() {
        let mut ring_buffer = FixedSizeRingBuffer::<_, 12>::default();
        for i in 0..100 {
            ring_buffer.push(i);
            assert_eq!(ring_buffer.iter().last(), Some(&i));
            println!("{i}");
        }
    }

    #[test]
    fn test_pushing_overflow() {
        let mut ring_buffer = FixedSizeRingBuffer::<_, 4>::default();
        ring_buffer.push("ce");
        assert_eq!(ring_buffer.iter().collect::<Vec<_>>(), vec![&"ce"]);
        ring_buffer.push("ll");
        assert_eq!(ring_buffer.iter().collect::<Vec<_>>(), vec![&"ce", &"ll"]);
        ring_buffer.push("ular");
        assert_eq!(
            ring_buffer.iter().collect::<Vec<_>>(),
            vec![&"ce", &"ll", &"ular"]
        );
        ring_buffer.push(" ");
        assert_eq!(
            ring_buffer.iter().collect::<Vec<_>>(),
            vec![&"ce", &"ll", &"ular", &" "]
        );
        ring_buffer.push("raza");
        assert_eq!(
            ring_buffer.iter().collect::<Vec<_>>(),
            vec![&"ll", &"ular", &" ", &"raza"]
        );
    }

    #[test]
    fn test_clone_full() {
        let mut ring_buffer = FixedSizeRingBuffer::<_, 4>::default();
        ring_buffer.push(1_usize);
        ring_buffer.push(2);
        ring_buffer.push(3);
        ring_buffer.push(4);
        let new_ring_buffer = ring_buffer.clone();
        assert_eq!(
            ring_buffer.iter().collect::<Vec<_>>(),
            new_ring_buffer.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_clone_partial() {
        let mut ring_buffer = FixedSizeRingBuffer::<_, 87>::default();
        for i in 0..100 {
            ring_buffer.push(i);
            let new_ring_buffer = ring_buffer.clone();
            assert_eq!(
                ring_buffer.iter().collect::<Vec<_>>(),
                new_ring_buffer.iter().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_serialize_full() {
        let mut ring_buffer = FixedSizeRingBuffer::<_, 4>::default();
        ring_buffer.push(1_u128);
        ring_buffer.push(2);
        ring_buffer.push(55);
        ring_buffer.push(12999);

        let serialized = serde_json::to_string(&ring_buffer).unwrap();
        assert_eq!(serialized, "[1,2,55,12999]");
    }

    #[test]
    fn test_serialize_partially_filled() {
        let mut ring_buffer = FixedSizeRingBuffer::<_, 4>::default();
        ring_buffer.push(1_u128);
        ring_buffer.push(2);

        let serialized = serde_json::to_string(&ring_buffer).unwrap();
        assert_eq!(serialized, "[1,2]");
    }

    #[test]
    fn test_deserialize_full() {
        let ring_buffer_string = "[-3,2,1023,-112]";
        let ring_buffer: FixedSizeRingBuffer<i16, 4> =
            serde_json::de::from_str(ring_buffer_string).unwrap();
        assert_eq!(
            ring_buffer.iter().collect::<Vec<_>>(),
            vec![&-3, &2, &1023, &-112]
        );
    }

    #[test]
    fn test_deserialize_partially_filled() {
        for i in 0..50 {
            let ring_buffer_values: Vec<_> = (0..i).collect();
            let string = format!("{:?}", ring_buffer_values);
            let ring_buffer: FixedSizeRingBuffer<_, 100> =
                serde_json::de::from_str(&string).unwrap();
            assert_eq!(ring_buffer.iter().collect::<Vec<_>>(), ring_buffer_values);
        }
    }
}

#[allow(unused)]
#[doc(hidden)]
mod test_derive_serde_ring_buffer {
    /// ```
    /// use serde::Serialize;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// #[derive(Serialize)]
    /// struct Something<T, const N: usize> {
    ///     ring_buffer: FixedSizeRingBuffer<T, N>,
    /// }
    /// ```
    fn derive_serialize() {}

    /// ```
    /// use serde::Deserialize;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// #[derive(Deserialize)]
    /// struct Something<T, const N: usize> {
    ///     ring_buffer: FixedSizeRingBuffer<T, N>,
    /// }
    /// ```
    fn derive_deserialize() {}

    /// ```
    /// use serde::{Deserialize, Serialize};
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// #[derive(Deserialize, Serialize)]
    /// struct Something<T, const N: usize> {
    ///     ring_buffer: FixedSizeRingBuffer<T, N>,
    /// }
    /// ```
    fn derive_serialize_deserialize() {}
}

#[allow(unused)]
#[doc(hidden)]
mod test_build_aux_storage {
    /// ```
    /// use serde::{Deserialize, Serialize};
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// build_aux_storage!(
    ///     name: __cr_AuxStorage,
    ///     aspects: [Mechanics]
    /// );
    /// let mut aux_storage = __cr_AuxStorage::<f32, f32, f32, f32, 4>::default();
    /// aux_storage.set_last_position(1_f32);
    /// aux_storage.set_last_position(3_f32);
    /// let last_positions = aux_storage.previous_positions().map(|f| *f).collect::<Vec<f32>>();
    /// assert_eq!(last_positions, vec![1_f32, 3_f32]);
    /// ```
    fn mechanics() {}

    /// ```
    /// use serde::{Deserialize, Serialize};
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// build_aux_storage!(
    ///     name: __cr_AuxStorage,
    ///     aspects: [Interaction]
    /// );
    /// let mut aux_storage = __cr_AuxStorage::default();
    /// aux_storage.incr_current_neighbours(1);
    /// aux_storage.incr_current_neighbours(2);
    /// aux_storage.incr_current_neighbours(1);
    /// assert_eq!(aux_storage.get_current_neighbours(), 4);
    /// ```
    fn interaction() {}

    /// ```
    /// use serde::{Deserialize, Serialize};
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// use cellular_raza_concepts::cycle::CycleEvent;
    /// build_aux_storage!(
    ///     name: __cr_AuxStorage,
    ///     aspects: [Cycle]
    /// );
    /// let mut aux_storage = __cr_AuxStorage::default();
    /// aux_storage.add_cycle_event(CycleEvent::Division);
    /// assert_eq!(aux_storage.get_cycle_events(), vec![CycleEvent::Division]);
    /// ```
    fn cycle() {}

    /// ```
    /// use serde::{Deserialize, Serialize};
    /// use cellular_raza_core_derive::*;
    /// use cellular_raza_core::backend::chili::aux_storage::*;
    /// build_aux_storage!(
    ///     name: __cr_AuxStorage,
    ///     aspects: [Reactions]
    /// );
    /// let mut aux_storage = __cr_AuxStorage::default();
    /// aux_storage.set_conc(0_f32);
    /// aux_storage.incr_conc(1.44_f32);
    /// assert_eq!(aux_storage.get_conc(), 0_f32 + 1.44_f32);
    /// ```
    fn reactions() {}
}
