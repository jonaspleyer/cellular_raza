use cellular_raza_concepts::CycleEvent;
pub use circ_buffer::*;
use serde::{Deserialize, Serialize};

pub use cellular_raza_concepts::Xapy;

// --------------------------------- UPDATE-MECHANICS --------------------------------
/// Used to store intermediate information about last positions and velocities.
/// Can store up to `N` values.
pub trait UpdateMechanics<Pos, Vel, For, const N: usize> {
    /// Stores the last position of the cell. May overwrite old results depending on
    /// how many old results are being stored.
    fn set_last_position(&mut self, pos: Pos);

    /// Get all previous positions. This number maybe smaller than the maximum number of stored
    /// positions but never exceeds it.
    fn previous_positions<'a>(&'a self) -> RingBufferIterRef<'a, Pos, N>;

    /// Stores the last velocity of the cell. Overwrites old results when stored amount
    /// exceeds number of maximum stored values.
    fn set_last_velocity(&mut self, vel: Vel);

    /// Get all previous velocities. This number may be smaller than the maximum number of stored
    /// velocities but never exceeds it.
    fn previous_velocities<'a>(&'a self) -> RingBufferIterRef<'a, Vel, N>;

    /// Get the number of previous values currently stored
    ///
    /// This number is by definition between 0 and `N`.
    fn n_previous_values(&self) -> usize;

    /// Add force to currently stored forces
    fn add_force(&mut self, force: For);

    /// Obtain current force on cell
    fn get_current_force_and_reset(&mut self) -> For;
}

/// Stores intermediate information about the mechanics of a cell.
#[derive(Clone, Deserialize, Serialize)]
pub struct AuxStorageMechanics<Pos, Vel, For, const N: usize> {
    positions: RingBuffer<Pos, N>,
    velocities: RingBuffer<Vel, N>,
    current_force: For,
    zero_force: For,
}

// It is necessary to implement this trait by hand since with the current version of the Mechanics
// concept, we need to specify next_random_mechanics_update: Some(0.0) in order for any updates to
// be done at all.
impl<Pos, Vel, For, const N: usize> Default for AuxStorageMechanics<Pos, Vel, For, N>
where
    For: num::Zero,
{
    fn default() -> Self {
        Self {
            positions: RingBuffer::<Pos, N>::default(),
            velocities: RingBuffer::<Vel, N>::default(),
            current_force: num::Zero::zero(),
            zero_force: num::Zero::zero(),
        }
    }
}

/// Used to construct initial (empty) AuxStorage variants.
pub trait DefaultFrom<T> {
    /// Constructs the Type in question from a given value. This is typically a zero value.
    /// If it can be constructed from the [num::Zero] trait, this method is not required and
    /// `cellular_raza` will determine the initial zero-value correctly.
    /// For other types (ie. dynamically-sized ones) additional entries may be necessary.
    fn default_from(value: &T) -> Self;
}

impl<Pos, Vel, For, const N: usize> DefaultFrom<For> for AuxStorageMechanics<Pos, Vel, For, N>
where
    For: Clone,
{
    fn default_from(value: &For) -> Self {
        let force: For = value.clone().into();
        Self {
            positions: RingBuffer::default(),
            velocities: RingBuffer::default(),
            current_force: force.clone(),
            zero_force: force.clone(),
        }
    }
}

impl<Pos, Vel, For, const N: usize> UpdateMechanics<Pos, Vel, For, N>
    for AuxStorageMechanics<Pos, Vel, For, N>
where
    For: Clone + core::ops::AddAssign<For>,
{
    #[inline]
    fn previous_positions<'a>(&'a self) -> RingBufferIterRef<'a, Pos, N> {
        self.positions.iter()
    }

    #[inline]
    fn previous_velocities<'a>(&'a self) -> RingBufferIterRef<'a, Vel, N> {
        self.velocities.iter()
    }

    #[inline]
    fn n_previous_values(&self) -> usize {
        self.positions.get_size()
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
    fn get_current_force_and_reset(&mut self) -> For {
        let f = self.current_force.clone();
        self.current_force = self.zero_force.clone();
        f
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
    fn get_cycle_events(&self) -> &Vec<CycleEvent>;

    /// Drain all cycle events
    fn drain_cycle_events<'a>(&'a mut self) -> std::vec::Drain<'a, CycleEvent>;

    /// Add another cycle event to the storage.
    fn add_cycle_event(&mut self, event: CycleEvent);
}

/// Stores intermediate information about the cell cycle.
///
/// This struct is used in the [build_aux_storage](crate::backend::chili::build_aux_storage) macro.
/// It can in principle also be re-used on its own since it implements the [UpdateCycle] trait.
///
/// ```
/// use cellular_raza_core::backend::chili::{AuxStorageCycle,UpdateCycle};
/// use cellular_raza_concepts::CycleEvent;
///
/// // Construct a new empty AuxStorageCycle
/// let mut aux_storage_cycle = AuxStorageCycle::default();
///
/// // Add one element
/// aux_storage_cycle.add_cycle_event(CycleEvent::Division);
/// assert_eq!(aux_storage_cycle.get_cycle_events().len(), 1);
///
/// // Drain all elements currently present
/// let events = aux_storage_cycle.drain_cycle_events();
/// assert_eq!(events.len(), 1);
/// ```
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
    fn get_cycle_events(&self) -> &Vec<CycleEvent> {
        &self.cycle_events
    }

    #[inline]
    fn drain_cycle_events<'a>(&'a mut self) -> std::vec::Drain<'a, CycleEvent> {
        self.cycle_events.drain(..)
    }

    #[inline]
    fn add_cycle_event(&mut self, event: CycleEvent) {
        self.cycle_events.push(event);
    }
}

// --------------------------------- UPDATE-REACTIONS --------------------------------
/// Interface to store intermediate information about cellular reactions.
pub trait UpdateReactions<Ri> {
    /// Set the value of intracellular concentrations
    fn set_conc(&mut self, conc: Ri);
    /// Obtain the current value of intracellular concentrations
    fn get_conc(&self) -> Ri;
    /// Add concentrations to the current value
    fn incr_conc(&mut self, incr: Ri);
}

/// Helper storage for values regarding intracellular concentrations for the
/// [Reactions](cellular_raza_concepts::Reactions) trait.
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct AuxStorageReactions<Ri> {
    concentration: Ri,
}

impl<Ri> DefaultFrom<Ri> for AuxStorageReactions<Ri>
where
    Ri: Clone,
{
    fn default_from(value: &Ri) -> Self {
        AuxStorageReactions {
            concentration: value.clone(),
        }
    }
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

/// Used to update properties of the cell related to the
/// [ReactionsContact](cellular_raza_concepts::ReactionsContact) trait.
pub trait UpdateReactionsContact<Ri, const N: usize> {
    /// Sets the current contact reactions increment
    fn set_current_increment(&mut self, new_increment: Ri);
    /// Adds to the current increment
    fn incr_current_increment(&mut self, increment: Ri);
    /// Obtains the current increment
    fn get_current_increment(&self) -> Ri;
    /// Obtain previous increments used for adams_bashforth integrators
    fn previous_increments<'a>(&'a self) -> RingBufferIterRef<'a, Ri, N>;
    /// Set the last increment in the ring buffer
    fn set_last_increment(&mut self, increment: Ri);
    /// Get the number of previous values to match against [circ_buffer::RingBufferIterRef]
    fn n_previous_values(&self) -> usize;
}

/// Implementor of the [UpdateReactionsContact] trait.
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct AuxStorageReactionsContact<Ri, const N: usize> {
    current_increment: Ri,
    increments: RingBuffer<Ri, N>,
}

impl<Ri, const N: usize> DefaultFrom<Ri> for AuxStorageReactionsContact<Ri, N>
where
    Ri: Clone,
{
    fn default_from(value: &Ri) -> Self {
        AuxStorageReactionsContact {
            current_increment: value.clone(),
            increments: Default::default(),
        }
    }
}

impl<Ri, const N: usize> UpdateReactionsContact<Ri, N> for AuxStorageReactionsContact<Ri, N>
where
    Ri: Clone + core::ops::Add<Ri, Output = Ri>,
{
    #[inline]
    fn get_current_increment(&self) -> Ri {
        self.current_increment.clone()
    }

    #[inline]
    fn incr_current_increment(&mut self, increment: Ri) {
        self.current_increment = self.current_increment.clone() + increment;
    }

    #[inline]
    fn set_current_increment(&mut self, new_increment: Ri) {
        self.current_increment = new_increment;
    }

    #[inline]
    fn previous_increments<'a>(&'a self) -> RingBufferIterRef<'a, Ri, N> {
        self.increments.iter()
    }

    #[inline]
    fn set_last_increment(&mut self, increment: Ri) {
        self.increments.push(increment)
    }

    #[inline]
    fn n_previous_values(&self) -> usize {
        self.increments.get_size()
    }
}

// -------------------------------- UPDATE-Interaction -------------------------------
/// Interface to store intermediate information about interactions.
pub trait UpdateInteraction {
    /// Obtain current number of neighbors
    fn get_current_neighbors(&self) -> usize;
    /// Set the number of neighbors
    fn set_current_neighbors(&mut self, neighbors: usize);
    /// Increment the number of current neighbors by the provided value
    fn incr_current_neighbors(&mut self, neighbors: usize);
}

/// Helper storage for number of neighbors of
/// [Interaction](cellular_raza_concepts::Interaction) trait.
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct AuxStorageInteraction {
    neighbor_count: usize,
}

impl UpdateInteraction for AuxStorageInteraction {
    #[inline]
    fn get_current_neighbors(&self) -> usize {
        self.neighbor_count
    }

    #[inline]
    fn incr_current_neighbors(&mut self, neighbors: usize) {
        self.neighbor_count += neighbors;
    }

    #[inline]
    fn set_current_neighbors(&mut self, neighbors: usize) {
        self.neighbor_count = neighbors;
    }
}

#[allow(unused)]
#[doc(hidden)]
mod test_derive_aux_storage_compile {
    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructMechanics<Pos, Vel, For, const N: usize> {
    ///     #[UpdateMechanics(Pos, Vel, For, N)]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, N>,
    /// }
    /// ```
    fn mechanics_default() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// pub struct TestStructMechanics<Pos, Vel, For, const N: usize> {
    ///     #[UpdateMechanics(Pos, Vel, For, N)]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, N>,
    /// }
    /// ```
    fn mechanics_visibility_1() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// pub(crate) struct TestStructMechanics<Pos, Vel, For, const N: usize> {
    ///     #[UpdateMechanics(Pos, Vel, For, N)]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, N>,
    /// }
    /// ```
    fn mechanics_visibility_2() {}

    /// ```
    /// mod some_module {
    ///     use cellular_raza_core::backend::chili::AuxStorage;
    ///     use cellular_raza_core::backend::chili::*;
    ///
    ///     #[derive(AuxStorage)]
    ///     pub(super) struct TestStructMechanics<Pos, Vel, For, const N: usize> {
    ///         #[UpdateMechanics(Pos, Vel, For, N)]
    ///         aux_mechanics: AuxStorageMechanics<Pos, Vel, For, N>,
    ///     }
    /// }
    /// fn use_impl<T, Pos, Vel, For, const N: usize>(
    ///     mut aux_storage: T
    /// ) -> For
    /// where
    ///     T: cellular_raza_core::backend::chili::UpdateMechanics<Pos, Vel, For, N>,
    /// {
    ///     aux_storage.get_current_force_and_reset()
    /// }
    /// ```
    fn mechanics_visibility_3() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructMechanics<Pos, Vel, For, T, const N: usize> {
    ///     #[UpdateMechanics(Pos, Vel, For, N)]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, N>,
    ///     other: T,
    /// }
    /// ```
    fn mechanics_more_struct_generics() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructMechanics<Pos, Vel, For, const N: usize, const M: usize> {
    ///     #[UpdateMechanics(Pos, Vel, For, N)]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, N>,
    ///     count: [i64; M],
    /// }
    /// ```
    fn mechanics_more_struct_const_generics() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructMechanics<Pos, Vel, For, const N: usize>
    /// where
    ///     Pos: Clone,
    /// {
    ///     #[UpdateMechanics(Pos, Vel, For, N)]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, N>,
    /// }
    /// ```
    fn mechanics_where_clause() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructMechanics<Pos, Vel, For, const N: usize> {
    ///     #[UpdateMechanics(Pos, Vel, For, N)]
    ///     #[cfg(not(test))]
    ///     aux_mechanics: AuxStorageMechanics<Pos, Vel, For, N>,
    /// }
    /// ```
    fn mechanics_other_attributes() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    /// use cellular_raza_concepts::CycleEvent;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructCycle {
    ///     #[UpdateCycle]
    ///     aux_cycle: AuxStorageCycle,
    /// }
    /// ```
    fn cycle_default() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    /// use cellular_raza_concepts::CycleEvent;
    ///
    /// #[derive(AuxStorage)]
    /// pub struct TestStructCycle {
    ///     #[UpdateCycle]
    ///     aux_cycle: AuxStorageCycle,
    /// }
    /// ```
    fn cycle_visibility_1() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    /// use cellular_raza_concepts::CycleEvent;
    ///
    /// #[derive(AuxStorage)]
    /// pub(crate) struct TestStructCycle {
    ///     #[UpdateCycle]
    ///     aux_cycle: AuxStorageCycle,
    /// }
    /// ```
    fn cycle_visibility_2() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    /// use cellular_raza_concepts::CycleEvent;
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
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    /// use cellular_raza_concepts::CycleEvent;
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
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    /// use cellular_raza_concepts::CycleEvent;
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
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    /// use cellular_raza_concepts::CycleEvent;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructCycle {
    ///     #[UpdateCycle]
    ///     #[cfg(not(test))]
    ///     aux_cycle: AuxStorageCycle,
    /// }
    /// ```
    fn cycle_other_attributes() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructReactions<R> {
    ///     #[UpdateReactions(R)]
    ///     aux_cycle: AuxStorageReactions<R>,
    /// }
    /// ```
    fn reactions_default() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// pub struct TestStructReactions<R> {
    ///     #[UpdateReactions(R)]
    ///     aux_cycle: AuxStorageReactions<R>,
    /// }
    /// ```
    fn reactions_visibility_1() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// pub(crate) struct TestStructReactions<R> {
    ///     #[UpdateReactions(R)]
    ///     aux_cycle: AuxStorageReactions<R>,
    /// }
    /// ```
    fn reactions_visibility_2() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
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
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
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
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
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
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructReactions<R> {
    ///     #[cfg(not(test))]
    ///     #[UpdateReactions(R)]
    ///     aux_cycle: AuxStorageReactions<R>,
    /// }
    /// ```
    fn reactions_other_attributes() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructInteraction {
    ///     #[UpdateInteraction]
    ///     aux_interaction: AuxStorageInteraction,
    /// }
    /// ```
    fn interactions_default() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// pub struct TestStructInteraction {
    ///     #[UpdateInteraction]
    ///     aux_interaction: AuxStorageInteraction,
    /// }
    /// ```
    fn interactions_visibility_1() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// pub(crate) struct TestStructInteraction {
    ///     #[UpdateInteraction]
    ///     aux_interaction: AuxStorageInteraction,
    /// }
    /// ```
    fn interactions_visibility_2() {}

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
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
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
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
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
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

    /// ```
    /// use cellular_raza_core::backend::chili::AuxStorage;
    /// use cellular_raza_core::backend::chili::*;
    ///
    /// #[derive(AuxStorage)]
    /// struct TestStructInteraction {
    ///     #[UpdateInteraction]
    ///     #[cfg(not(test))]
    ///     aux_interaction: AuxStorageInteraction,
    /// }
    /// ```
    fn interactions_other_attributes() {}
}

#[cfg(test)]
mod test_derive_aux_storage {
    use super::*;
    use cellular_raza_core_proc_macro::AuxStorage;

    #[derive(AuxStorage)]
    struct TestStructDouble<Pos, Vel, For, const N: usize> {
        #[UpdateCycle]
        aux_cycle: AuxStorageCycle,
        #[UpdateMechanics(Pos, Vel, For, N)]
        aux_mechanics: AuxStorageMechanics<Pos, Vel, For, N>,
    }

    #[derive(AuxStorage)]
    struct TestStructCycle {
        #[UpdateCycle]
        aux_cycle: AuxStorageCycle,
    }

    #[derive(AuxStorage)]
    struct TestStructMechanics<Pos, Vel, For, const N: usize> {
        #[UpdateMechanics(Pos, Vel, For, N)]
        aux_mechanics: AuxStorageMechanics<Pos, Vel, For, N>,
    }

    fn add_get_events<A>(aux_storage: &mut A)
    where
        A: UpdateCycle,
    {
        aux_storage.add_cycle_event(CycleEvent::Division);
        let events = aux_storage.get_cycle_events();
        assert_eq!(events, &vec![CycleEvent::Division]);
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
        assert_eq!(events, &initial_events);
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
        let mut aux_storage = TestStructMechanics::<_, _, _, 4> {
            aux_mechanics: AuxStorageMechanics::default(),
        };
        aux_storage.set_last_position(3_f64);
        aux_storage.set_last_velocity(5_f32);
        aux_storage.add_force(1_f32);
    }

    #[test]
    fn cycle_mechanics_add_get_events() {
        let mut aux_storage = TestStructDouble::<_, _, _, 4> {
            aux_cycle: AuxStorageCycle::default(),
            aux_mechanics: AuxStorageMechanics::default(),
        };
        aux_storage.set_last_position(3_f64);
        aux_storage.set_last_velocity(5_f32);
        aux_storage.add_force(-5_f32);
        add_get_events(&mut aux_storage);
    }

    #[test]
    fn cycle_mechanics_set_get_events() {
        let mut aux_storage = TestStructDouble::<_, _, _, 4> {
            aux_cycle: AuxStorageCycle::default(),
            aux_mechanics: AuxStorageMechanics::default(),
        };
        aux_storage.set_last_position(3_f64);
        aux_storage.set_last_velocity(5_f32);
        aux_storage.add_force(111_i64);
        set_get_events(&mut aux_storage);
    }
}

#[allow(unused)]
#[doc(hidden)]
mod test_build_aux_storage {
    use crate::backend::chili::proc_macro::aux_storage_constructor;
    macro_rules! construct (
        (name:$test_name:ident,
        aspects:[$($asp:ident),*]) => {
            /// ```
            /// use serde::{Deserialize, Serialize};
            /// use cellular_raza_core::backend::chili::*;
            /// build_aux_storage!(
            #[doc = concat!("aspects: [", $(stringify!($asp,),)* "],")]
            ///     aux_storage_name: __cr_AuxStorage,
            ///     core_path: cellular_raza_core
            /// );
            // #[doc = concat!("let mut aux_storage = __cr_AuxStorage {", init!($($asp),*) "};")]
            // #[doc = init!{@start $($asp),* end}]
            #[doc = concat!(
                "let mut aux_storage = (",
                stringify!(aux_storage_constructor!(
                    aux_storage_name: __cr_AuxStorage,
                    core_path: cellular_raza_core,
                    aspects: [$($asp),*],
                )),
                ")(());",
            )]
            /// macro_rules! test_aspect (
            ///     (Mechanics) => {
            ///         {
            ///             use cellular_raza_core::backend::chili::UpdateMechanics;
            ///             aux_storage.set_last_position(1_f32);
            ///             aux_storage.set_last_position(3_f32);
            ///             let last_positions = aux_storage
            ///                 .previous_positions()
            ///                 .map(|f| *f)
            ///                 .collect::<Vec<f32>>();
            ///             assert_eq!(last_positions, vec![1_f32, 3_f32]);
            ///             aux_storage.set_last_velocity(10_f32);
            ///             let last_velocities: cellular_raza_core::backend::chili::RingBufferIterRef<_, 4>
            ///                 = aux_storage.previous_velocities();
            ///             let last_velocities = last_velocities.map(|f| *f).collect::<Vec<f32>>();
            ///             assert_eq!(last_velocities, vec![10_f32]);
            ///             aux_storage.add_force(22_f32);
            ///             assert_eq!(aux_storage.get_current_force_and_reset(), 22_f32);
            ///         }
            ///     };
            ///     (Interaction) => {
            ///         {
            ///             use cellular_raza_core::backend::chili::UpdateInteraction;
            ///             aux_storage.incr_current_neighbors(1);
            ///             aux_storage.incr_current_neighbors(2);
            ///             aux_storage.incr_current_neighbors(1);
            ///             assert_eq!(aux_storage.get_current_neighbors(), 4);
            ///         }
            ///     };
            ///     (Cycle) => {
            ///         {
            ///             use cellular_raza_core::backend::chili::UpdateCycle;
            ///             use cellular_raza_concepts::CycleEvent;
            ///             aux_storage.add_cycle_event(CycleEvent::Division);
            ///             assert_eq!(aux_storage.get_cycle_events(), &vec![CycleEvent::Division]);
            ///         }
            ///     };
            ///     (Reactions) => {
            ///         {
            ///             use cellular_raza_core::backend::chili::UpdateReactions;
            ///             aux_storage.set_conc(0_f32);
            ///             aux_storage.incr_conc(1.44_f32);
            ///             assert_eq!(aux_storage.get_conc(), 0_f32 + 1.44_f32);
            ///         }
            ///     };
            ///     (ReactionsContact) => {
            ///         {
            ///             use cellular_raza_core::backend::chili::UpdateReactionsContact;
            ///             aux_storage.set_last_increment(0f32);
            ///             aux_storage.set_last_increment(3f32);
            ///             assert_eq!(UpdateReactionsContact::n_previous_values(&aux_storage), 2);
            ///             let last_increments =
            ///                 UpdateReactionsContact::<f32, 10>::previous_increments(
            ///                 &aux_storage
            ///             );
            ///             let last_increments = last_increments.map(|f| *f).collect::<Vec<_>>();
            ///             assert_eq!(last_increments, vec![0.0, 3.0]);
            ///         }
            ///     };
            /// );
            #[doc = concat!($(
                concat!("test_aspect!(", stringify!($asp), ");")
            ,)*)]
            /// ```
            fn $test_name() {}
        }
    );

    cellular_raza_core_proc_macro::run_test_for_aspects!(
        test: construct,
        aspects: [Mechanics, Interaction, Cycle, Reactions, ReactionsContact]
    );
}
