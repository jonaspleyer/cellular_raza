use cellular_raza_concepts::{errors::CalcError, prelude::IndexError};
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::{errors::SimulationError, VoxelPlainIndex};

///
/// This very simple implementation uses the [Barrier](hurdles::Barrier) struct from the [hurdles] crate
/// which should in theory perform faster than the [std::sync::Barrier] struct from the standard library.
///
/// By using the [SyncSubDomains] trait, we can automatically create a collection of syncers
/// which can then be simply given to the respective threads and handle synchronization.
/// ```
/// # use std::collections::HashMap;
/// # use cellular_raza_core::backend::chili::simulation_flow::{BarrierSync, FromMap, SyncSubDomains};
/// let map = HashMap::from_iter([
///     (0, vec![1]),
///     (1, vec![0]),
/// ]);
/// let mut syncers = BarrierSync::from_map(&map).unwrap();
/// assert_eq!(syncers.len(), 2);
///
/// let mut syncer_0 = syncers.remove(&0).unwrap();
/// let mut syncer_1 = syncers.remove(&1).unwrap();
///
/// // Define the number of iterations to run
/// let n_iterations = 10;
/// let shared_counter = std::sync::Arc::new(std::sync::Mutex::new(0_i64));
///
/// let shared_counter_0 = std::sync::Arc::clone(&shared_counter);
/// let handle_0 = std::thread::spawn(move || {
///     for _ in 0..n_iterations {
///         syncer_0.sync();
///         *shared_counter_0.lock().unwrap() += 1;
///         syncer_0.sync();
///     }
/// });
///
/// for i in 0..n_iterations {
///     syncer_1.sync();
///     syncer_1.sync();
///     assert_eq!(*shared_counter.lock().unwrap(), i+1);
/// }
/// handle_0.join();
/// ```
pub struct BarrierSync {
    barrier: hurdles::Barrier,
}

/// Constructs a collection of Items from a map (graph)
pub trait FromMap<I>
where
    Self: Sized,
{
    /// [SubDomains](super::concepts::SubDomain) can be neighboring each other via complicated graphs.
    /// An easy way to represent this is by using a [HashMap]. We want to create Barriers which match
    /// the specified subdomain indices.
    fn from_map(map: &std::collections::HashMap<I, Vec<I>>) -> Result<HashMap<I, Self>, IndexError>
    where
        I: Eq + core::hash::Hash + Clone;
}

/// Responsible for syncing the simulation between different threads.
pub trait SyncSubDomains {
    /// Function which forces connected syncers to wait for each other.
    /// This approach does not necessarily require all threads to wait but can mean that
    /// only depending threads wait for each other.
    fn sync(&mut self);
}

impl<I> FromMap<I> for BarrierSync {
    fn from_map(map: &std::collections::HashMap<I, Vec<I>>) -> Result<HashMap<I, Self>, IndexError>
    where
        I: Eq + core::hash::Hash + Clone,
    {
        let barrier = hurdles::Barrier::new(map.len());
        let res = map
            .keys()
            .map(|i| {
                (
                    i.clone(),
                    Self {
                        barrier: barrier.clone(),
                    },
                )
            })
            .collect();
        Ok(res)
    }
}

impl SyncSubDomains for BarrierSync {
    fn sync(&mut self) {
        self.barrier.wait();
    }
}

/// A [TimeEvent] describes that a certain action is to be executed after the next iteration step.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub enum TimeEvent {
    /// Saves a partial simulation run which is suitable for data readout but not full recovery of the
    /// simulation for restarting.
    PartialSave,
    /// Performs a complete save from which the simulation should be able to be recovered.
    FullSave,
}

/// Increments time of the simulation
///
/// In the future we hope to add adaptive steppers depending on a specified accuracy function.
pub trait TimeStepper<F> {
    /// Advances the time stepper to the next time point. Also returns if there is an event
    /// scheduled to take place and the next time value and iteration number
    #[must_use]
    fn advance(&mut self) -> Result<Option<(F, i64, Option<TimeEvent>)>, CalcError>;

    /// Obtains the current time the simulation is at
    fn get_current_time(&self) -> Option<F>;

    /// Obtains the current number of iteration
    fn get_current_iteration(&self) -> Option<i64>;

    /// Obtains information about if a [TimeEvent] is scheduled to take place this iteration.
    fn get_current_event(&self) -> Option<TimeEvent>;

    /// Retrieved the last point at which the simulation was fully recovered.
    /// This might be helpful in the future when error handling is more mature and able to recover.
    fn get_last_full_save(&self) -> Option<(F, i64)>;
}

/// Time stepping with a fixed time length
///
/// This time-stepper increments the time variable by the same length.
/// ```
/// # use cellular_raza_core::backend::chili::simulation_flow::FixedStepsize;
/// let t0 = 1.0;
/// let dt = 0.2;
/// let save_points = vec![3.0, 5.0, 11.0, 20.0];
/// let time_stepper = FixedStepsize::from_save_points(t0, dt, save_points).unwrap();
/// ```
#[derive(Clone, Deserialize, Serialize)]
pub struct FixedStepsize<F> {
    // The stepsize which was fixed
    dt: F,
    t0: F,
    // An ordered set of time points to store every value at which we should evaluate
    all_events: Vec<(F, i64, TimeEvent)>,
    current_time: F,
    current_iteration: i64,
    maximum_iterations: i64,
    current_event: Option<TimeEvent>,
    past_events: Vec<(F, i64, TimeEvent)>,
}

impl<F> FixedStepsize<F>
where
    F: num::Float + num::ToPrimitive,
{
    /// Simple function to construct the stepper from an initial time point, the time increment and
    /// the time points at which the simulation should be saved. Notice that these saves do not cover
    /// [FullSaves](TimeEvent::FullSave) but only [PartialSaves](TimeEvent::PartialSave).
    pub fn from_save_points(t0: F, dt: F, save_points: Vec<F>) -> Result<Self, CalcError> {
        // Sort the save points
        let mut save_points = save_points;
        save_points.sort_by(|x, y| x.partial_cmp(y).unwrap());
        if save_points.iter().any(|x| t0 > *x) {
            return Err(CalcError(
                "Invalid time configuration! Evaluation time point is before starting time point."
                    .to_owned(),
            ));
        }
        let last_save_point = save_points
            .clone()
            .into_iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .ok_or(CalcError(
                "No savepoints specified. Simulation will not save any results.".to_owned(),
            ))?;
        let maximum_iterations =
            (((last_save_point - t0) / dt).round())
                .to_i64()
                .ok_or(CalcError(
                    "An error in casting of float type to i64 occurred".to_owned(),
                ))?;
        let all_events = save_points
            .clone()
            .into_iter()
            .map(|x| {
                (
                    x,
                    ((x - t0) / dt).round().to_i64().unwrap(),
                    TimeEvent::PartialSave,
                )
            })
            .collect();

        let current_event = if t0
            == save_points
                .into_iter()
                .min_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap()
        {
            Some(TimeEvent::PartialSave)
        } else {
            None
        };

        Ok(Self {
            dt,
            t0,
            all_events,
            current_time: t0,
            current_iteration: 0,
            maximum_iterations,
            // TODO check this again
            current_event,
            past_events: Vec::new(),
        })
    }
}

impl<F> TimeStepper<F> for FixedStepsize<F>
where
    F: num::Float + num::FromPrimitive,
{
    fn advance(&mut self) -> Result<Option<(F, i64, Option<TimeEvent>)>, CalcError> {
        self.current_iteration += 1;
        self.current_time = F::from_i64(self.current_iteration).ok_or(CalcError(
            "Error when casting from i64 to floating point value".to_owned(),
        ))? * self.dt
            + self.t0;
        // TODO Check if a current event should take place
        if let Some((_, _, event)) = self
            .all_events
            .iter()
            .filter(|(_, iteration, _)| *iteration == self.current_iteration)
            .last()
        {
            self.current_event = Some(*event);
        } else {
            self.current_event = None;
        }

        match (
            self.get_current_time(),
            self.get_current_iteration(),
            self.get_current_event(),
        ) {
            (Some(time), Some(iteration), event) => {
                match event {
                    Some(e) => self.past_events.push((time, iteration, e)),
                    _ => (),
                }
                Ok(Some((time, iteration, event)))
            }
            _ => Ok(None),
        }
    }

    fn get_current_time(&self) -> Option<F> {
        if self.current_iteration <= self.maximum_iterations {
            Some(self.current_time)
        } else {
            None
        }
    }

    fn get_current_iteration(&self) -> Option<i64> {
        if self.current_iteration <= self.maximum_iterations {
            Some(self.current_iteration)
        } else {
            None
        }
    }

    fn get_current_event(&self) -> Option<TimeEvent> {
        self.current_event
    }

    fn get_last_full_save(&self) -> Option<(F, i64)> {
        self.past_events
            .clone()
            .into_iter()
            .filter(|(_, _, event)| *event == TimeEvent::FullSave)
            .last()
            .and_then(|x| Some((x.0, x.1)))
    }
}

pub trait Communicator<I, T>
where
    Self: Sized,
{
    fn send(&mut self, receiver: &I, message: T) -> Result<(), SimulationError>;
    fn receive(&mut self) -> Vec<T>;
}

#[derive(Clone)]
pub struct ChannelComm<I, T> {
    senders: std::collections::BTreeMap<I, crossbeam_channel::Sender<T>>,
    receiver: crossbeam_channel::Receiver<T>,
}

impl<T, I> FromMap<I> for ChannelComm<I, T>
where
    I: Clone + core::hash::Hash + Eq + Ord,
{
    fn from_map(map: &HashMap<I, Vec<I>>) -> Result<HashMap<I, Self>, IndexError> {
        let channels: HashMap<_, _> = map
            .keys()
            .into_iter()
            .map(|key| {
                let (s, r) = crossbeam_channel::unbounded::<T>();
                (key, (s, r))
            })
            .collect();
        let mut comms = HashMap::new();
        for key in map.keys().into_iter() {
            let senders = map
                .get(&key)
                .ok_or(IndexError("Network of communicators could not be constructed due to incorrect entries in map".into()))?
                .clone()
                .into_iter()
                .map(|connected_key| (key.clone(), channels[&connected_key].0.clone())).collect();

            let comm = ChannelComm {
                senders,
                receiver: channels[&key].1.clone(),
            };
            comms.insert(key.clone(), comm);
        }
        Ok(comms)
    }
}

impl<I, T> Communicator<I, T> for ChannelComm<I, T>
where
    I: core::hash::Hash + Eq + Ord,
{
    fn receive(&mut self) -> Vec<T> {
        self.receiver.try_iter().collect()
    }

    fn send(&mut self, receiver: &I, message: T) -> Result<(), SimulationError> {
        self.senders[&receiver].send(message)?;
        Ok(())
    }
}

pub struct PosInformation<Pos, Vel, Inf> {
    pub pos: Pos,
    pub vel: Vel,
    pub info: Inf,
    pub count: usize,
    pub index_sender: VoxelPlainIndex,
    pub index_receiver: VoxelPlainIndex,
}

pub struct ForceInformation<For> {
    pub force: For,
    pub count: usize,
    pub index_sender: VoxelPlainIndex,
}

pub struct SendCell<Cel, Aux>(pub Cel, pub Aux);

#[doc(hidden)]
#[allow(unused)]
mod test_derive_communicator {
    /// ```
    /// use cellular_raza_core::derive::Communicator;
    /// use cellular_raza_core::backend::chili::{
    ///     errors::SimulationError,
    ///     simulation_flow::{ChannelComm, Communicator}
    /// };
    /// #[derive(Communicator)]
    /// struct MyComm<I, T> {
    ///     #[Comm(I, T)]
    ///     comm: ChannelComm<I, T>
    /// }
    /// ```
    fn default() {}

    /// ```
    /// use cellular_raza_core::derive::Communicator;
    /// use cellular_raza_core::backend::chili::{
    ///     errors::SimulationError,
    ///     simulation_flow::{ChannelComm, Communicator}
    /// };
    /// #[derive(Communicator)]
    /// struct MyDouble<I> {
    ///     #[Comm(I, String)]
    ///     comm1: ChannelComm<I, String>,
    ///     #[Comm(I, f64)]
    ///     comm2: ChannelComm<I, f64>,
    /// }
    /// ```
    fn two_communicators_explicit() {}

    /// ```
    /// use cellular_raza_core::derive::Communicator;
    /// use cellular_raza_core::backend::chili::{
    ///     errors::SimulationError,
    ///     simulation_flow::{ChannelComm, Communicator}
    /// };
    /// struct Message<T>(T);
    /// #[derive(Communicator)]
    /// struct MyDouble<I, T> {
    ///     #[Comm(I, Message<T>)]
    ///     comm1: ChannelComm<I, Message<T>>,
    ///     #[Comm(I, f64)]
    ///     comm2: ChannelComm<I, f64>,
    /// }
    /// ```
    fn two_communicators_generic_one() {}
}

#[doc(hidden)]
#[allow(unused)]
mod test_build_communicator {
    macro_rules! test_build_communicator(
        (
            name:$func_name:ident,
            aspects:[$($asp:ident),*]
        ) => {
            /// ```
            /// use cellular_raza_core::derive::build_communicator;
            /// build_communicator!(
            ///     name: __MyComm,
            ///     aspects: [
            #[doc = stringify!($($asp),*)]
            ///     ],
            ///     simulation_flow_path: cellular_raza_core::backend::chili::simulation_flow,
            ///     core_path: cellular_raza_core
            /// );
            /// ```
            #[allow(non_snake_case)]
            fn $func_name () {}
        };
    );

    cellular_raza_core_derive::run_test_for_aspects!(
        test: test_build_communicator,
        aspects: [Mechanics, Interaction, Cycle, Reactions]
    );
}

#[cfg(test)]
pub mod test_sync {
    use super::*;
    use std::sync::*;

    fn test_single_map<S>(map: HashMap<usize, Vec<usize>>)
    where
        S: 'static + SyncSubDomains + FromMap<usize> + Send + Sync,
    {
        // Define the number of threads and iterations to use
        let n_iterations = 1_000;
        let n_threads = map.len();

        // We count the number of iterations via this mutex.
        // Individual threads will increment their counter by +1 each time they are executed
        let iteration_counter =
            Arc::new(Mutex::new(Vec::from_iter((0..n_threads).map(|_| 0_usize))));

        // Create a barrier from which we
        let syncers = S::from_map(&map).unwrap();
        let handles = syncers
            .into_iter()
            .map(|(n_thread, mut syncer)| {
                let iteration_counter_thread = Arc::clone(&iteration_counter);
                std::thread::spawn(move || {
                    for n_iteration in 0..n_iterations {
                        syncer.sync();
                        iteration_counter_thread.lock().unwrap()[n_thread] += 1;
                        syncer.sync();
                        let current_value = iteration_counter_thread.lock().unwrap().clone();
                        assert_eq!(current_value, vec![n_iteration + 1; n_threads]);
                    }
                })
            })
            .collect::<Vec<_>>();

        for handle in handles.into_iter() {
            handle.join().unwrap();
        }
    }

    fn test_multiple_maps<S>()
    where
        S: 'static + SyncSubDomains + FromMap<usize> + Send + Sync,
    {
        let map0 = HashMap::from_iter([(0, vec![1]), (1, vec![0])]);
        test_single_map::<S>(map0);

        let map1 = HashMap::from_iter([(0, vec![1, 2]), (1, vec![0, 2]), (2, vec![0, 1])]);
        test_single_map::<S>(map1);

        let map2 = HashMap::from_iter([
            (0, vec![1, 2, 3]),
            (1, vec![0, 2, 3]),
            (2, vec![0, 1, 3]),
            (3, vec![0, 1, 2]),
        ]);
        test_single_map::<S>(map2);

        let map3 = HashMap::from_iter([
            (0, vec![1, 2]),
            (1, vec![0, 3]),
            (2, vec![0, 3]),
            (3, vec![1, 2]),
        ]);
        test_single_map::<S>(map3);

        let map4 = HashMap::from_iter([
            (0, vec![1]),
            (1, vec![2]),
            (2, vec![3]),
            (3, vec![4]),
            (4, vec![0]),
        ]);
        test_single_map::<BarrierSync>(map4);
    }

    #[test]
    fn barrier_sync() {
        test_multiple_maps::<BarrierSync>();
    }
}

#[cfg(test)]
pub mod test_time_stepper {
    use rand::Rng;
    use rand::SeedableRng;

    use super::*;

    fn generate_new_fixed_stepper<F>(rng_seed: u64) -> FixedStepsize<F>
    where
        F: num::Float + From<f32>,
    {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(rng_seed);
        let t0 = <F as From<_>>::from(rng.gen_range(0.0..1.0));
        let dt = <F as From<_>>::from(rng.gen_range(0.1..2.0));
        let save_points = vec![
            <F as From<_>>::from(rng.gen_range(0.01..1.8)),
            <F as From<_>>::from(rng.gen_range(2.01..3.8)),
            <F as From<_>>::from(rng.gen_range(4.01..5.8)),
            <F as From<_>>::from(rng.gen_range(6.01..7.8)),
        ];
        FixedStepsize::<F>::from_save_points(t0, dt, save_points).unwrap()
    }

    #[test]
    fn initialization() {
        let t0 = 1.0;
        let dt = 0.2;
        let save_points = vec![3.0, 5.0, 11.0, 20.0];
        let time_stepper = FixedStepsize::from_save_points(t0, dt, save_points).unwrap();
        assert_eq!(Some(t0), time_stepper.get_current_time());
        assert_eq!(Some(0), time_stepper.get_current_iteration());
        assert_eq!(None, time_stepper.get_current_event());
    }

    #[test]
    #[should_panic]
    fn panic_wrong_save_points() {
        let t0 = 10.0;
        let dt = 0.2;
        let save_points = vec![3.0, 5.0, 11.0, 20.0];
        // This call should fail since t0 is larger than the first two save points
        let _time_stepper = FixedStepsize::from_save_points(t0, dt, save_points).unwrap();
    }

    #[test]
    fn stepping_1() {
        let t0 = 1.0;
        let dt = 0.2;
        let save_points = vec![3.0, 5.0, 11.0, 20.0];
        let mut time_stepper = FixedStepsize::from_save_points(t0, dt, save_points).unwrap();

        assert_eq!(Some(t0), time_stepper.get_current_time());
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + dt, 1_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 2.0 * dt, 2_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 3.0 * dt, 3_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 4.0 * dt, 4_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 5.0 * dt, 5_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 6.0 * dt, 6_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 7.0 * dt, 7_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 8.0 * dt, 8_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 9.0 * dt, 9_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(
            Some((t0 + 10.0 * dt, 10_i64, Some(TimeEvent::PartialSave))),
            next
        );
    }

    #[test]
    fn stepping_2() {
        let t0 = 0.0;
        let dt = 0.1;
        let save_points = vec![0.5, 0.7, 0.9, 1.0];
        let mut time_stepper = FixedStepsize::from_save_points(t0, dt, save_points).unwrap();

        assert_eq!(Some(t0), time_stepper.get_current_time());
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + dt, 1_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 2.0 * dt, 2_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 3.0 * dt, 3_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 4.0 * dt, 4_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(
            Some((t0 + 5.0 * dt, 5_i64, Some(TimeEvent::PartialSave))),
            next
        );
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 6.0 * dt, 6_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(
            Some((t0 + 7.0 * dt, 7_i64, Some(TimeEvent::PartialSave))),
            next
        );
        let next = time_stepper.advance().unwrap();
        assert_eq!(Some((t0 + 8.0 * dt, 8_i64, None)), next);
        let next = time_stepper.advance().unwrap();
        assert_eq!(
            Some((t0 + 9.0 * dt, 9_i64, Some(TimeEvent::PartialSave))),
            next
        );
        let next = time_stepper.advance().unwrap();
        assert_eq!(
            Some((t0 + 10.0 * dt, 10_i64, Some(TimeEvent::PartialSave))),
            next
        );
    }

    fn test_stepping(rng_seed: u64) {
        let mut time_stepper = generate_new_fixed_stepper::<f32>(rng_seed);

        for _ in 0..100 {
            let res = time_stepper.advance().unwrap();
            match res {
                Some(_) => (),
                None => return,
            }
        }
        panic!("The time stepper should have reached the end by now");
    }

    #[test]
    fn stepping_end_0() {
        test_stepping(0);
    }

    #[test]
    fn stepping_end_1() {
        test_stepping(1);
    }

    #[test]
    fn stepping_end_2() {
        test_stepping(2);
    }

    #[test]
    fn stepping_end_3() {
        test_stepping(3);
    }
}
