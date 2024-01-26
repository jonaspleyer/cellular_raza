use cellular_raza_concepts::{CalcError, IndexError};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use std::{collections::HashMap, marker::PhantomData};

use super::{errors::SimulationError, VoxelPlainIndex};

///
/// This very simple implementation uses the [hurdles::Barrier] struct which should
/// in theory perform faster than the [std::sync::Barrier] struct from the standard library.
///
/// By using the [SyncSubDomains] trait, we can automatically create a collection of syncers
/// which can then be simply given to the respective threads and handle synchronization.
/// ```
/// # use std::collections::HashMap;
/// # use cellular_raza_core::backend::chili::{BarrierSync, FromMap, SyncSubDomains};
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

/// Validates a given map.
///
/// This algorithm checks if every keys neighbours also contain the specified key.
/// If this is not the case, the map cannot be considered valid.
/// Note that this algorithm does not check if all keys are connected.
/// This means, disjoint sets are allowed.
///
/// ```
/// use cellular_raza_core::backend::chili::validate_map;
///
/// let new_map = std::collections::HashMap::from([
///     (1_usize, vec![0,2]),
///     (2_usize, vec![1,3]),
///     (3_usize, vec![2,0]),
///     (0_usize, vec![3,1]),
/// ]);
///
/// let is_valid = validate_map(&new_map);
/// assert_eq!(is_valid, true);
/// ```
pub fn validate_map<I>(map: &std::collections::HashMap<I, Vec<I>>) -> bool
where
    I: Eq + core::hash::Hash + Clone + Ord,
{
    for (index, neighbours) in map.iter() {
        if neighbours.iter().any(|n| match map.get(n) {
            Some(reverse_neighbours) => !reverse_neighbours.contains(index),
            None => true,
        }) {
            return false;
        }
    }
    true
}

/// Undirected graph
///
/// This datatype is currently only used to create the simulation subdomains.
/// There are no plans to extend the use of this object.
pub struct UDGraph<I>(pub(crate) Vec<(I, I)>);

impl<I> UDGraph<I> {
    /// Construct a new undirected graph.
    ///
    /// ```
    /// # use cellular_raza_core::backend::chili::UDGraph;
    /// let ud_graph: UDGraph<usize> = UDGraph::new();
    /// ```
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Push an additional connection between nodes to the UDGraph
    ///
    /// This will return an option in either of two cases
    /// 1. The connection is already contained in the Graph
    /// 2. The nodes are identical, ie. `new_connection.0==new_connection.1`
    ///
    /// ```
    /// # use cellular_raza_core::backend::chili::UDGraph;
    /// let mut ud_graph = UDGraph::new();
    /// assert_eq!(ud_graph.push((1, 2)), None);
    /// assert_eq!(ud_graph.push((1, 3)), None);
    ///
    /// // These cases will not return `None` due to the
    /// // specified reasons above
    /// assert_eq!(ud_graph.push((1, 2)), Some((1, 2)));
    /// assert_eq!(ud_graph.push((1, 1)), Some((1, 1)));
    /// ```
    pub fn push(&mut self, new_connection: (I, I)) -> Option<(I, I)>
    where
        I: PartialEq,
    {
        if new_connection.0 == new_connection.1 {
            return Some(new_connection);
        }
        if self
            .0
            .iter()
            .any(|connection| connection == &new_connection)
        {
            return Some(new_connection);
        }
        self.0.push(new_connection);
        None
    }

    /// Extends the [UDGraph] with new connections
    ///
    /// Will return all connections which can not be added by the [push](UDGraph::push) method.
    ///
    /// ```
    /// # use cellular_raza_core::backend::chili::UDGraph;
    /// let mut ud_graph = UDGraph::new();
    /// let new_connections = [
    ///     (1_f64, 2_f64),
    ///     (2_f64, 3_f64),
    ///     (3_f64, 4_f64),
    /// ];
    /// let res = ud_graph.extend(new_connections);
    /// assert_eq!(res, vec![]);
    /// ```
    pub fn extend<J>(&mut self, new_connections: J) -> Vec<(I, I)>
    where
        I: PartialEq,
        J: IntoIterator<Item = (I, I)>,
    {
        new_connections
            .into_iter()
            .filter_map(|new_connection| self.push(new_connection))
            .collect()
    }

    /// Clears the [UDGraph] thus removing all connections.
    ///
    /// See [std::vec::Vec::clear].
    pub fn clear(&mut self) {
        self.0.clear()
    }

    /// Drains the [UDGraph], thus returning an iterator over the specified elements.
    ///
    /// See [std::vec::Vec::drain].
    pub fn drain<R>(&mut self, range: R) -> std::vec::Drain<'_, (I, I)>
    where
        R: core::ops::RangeBounds<usize>,
    {
        self.0.drain(range)
    }

    /// Returns all nodes currently stored in the [UDGraph].
    ///
    /// ```
    /// # use cellular_raza_core::backend::chili::UDGraph;
    /// let mut ud_graph = UDGraph::new();
    /// ud_graph.push(("a", "s"));
    /// ud_graph.push(("a", "K"));
    /// ud_graph.push(("h", "s"));
    ///
    /// assert_eq!(ud_graph.nodes(), vec![&"a", &"s", &"K", &"h"]);
    /// ```
    pub fn nodes(&self) -> Vec<&I>
    where
        I: Clone + Eq + core::hash::Hash,
    {
        self.0
            .iter()
            .map(|(c1, c2)| [c1, c2].into_iter())
            .flatten()
            .unique()
            .collect()
    }
}

impl<I> IntoIterator for UDGraph<I> {
    type Item = (I, I);
    type IntoIter = std::vec::IntoIter<(I, I)>;

    /// Consumes the graph and returns iterator over elements.
    ///
    /// See [std::vec::Vec::into_iter].
    fn into_iter(self) -> std::vec::IntoIter<(I, I)> {
        self.0.into_iter()
    }
}

impl<I> UDGraph<I>
where
    I: core::hash::Hash + Clone + Eq,
{
    /// Convert the [UDGraph] into a regular [HashMap].
    ///
    /// ```
    /// # use cellular_raza_core::backend::chili::UDGraph;
    /// let mut ud_graph = UDGraph::new();
    /// ud_graph.push((1, 2));
    /// ud_graph.push((2, 3));
    /// ud_graph.push((3, 1));
    ///
    /// let map = ud_graph.to_hash_map();
    /// assert_eq!(map.keys().len(), 3);
    /// ```
    pub fn to_hash_map(self) -> std::collections::HashMap<I, Vec<I>> {
        let mut map: HashMap<_, Vec<I>> = self
            .0
            .iter()
            .map(|x| [(x.0.clone(), Vec::new()), (x.1.clone(), Vec::new())].into_iter())
            .flatten()
            .collect();
        self.0.iter().for_each(|(c1, c2)| {
            map.entry(c1.clone()).and_modify(|v| v.push(c2.clone()));
            map.entry(c2.clone()).and_modify(|v| v.push(c1.clone()));
        });
        map
    }
}

impl<I> core::ops::Deref for UDGraph<I> {
    type Target = Vec<(I, I)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<I> From<UDGraph<I>> for PhantomData<I> {
    #[allow(unused)]
    fn from(value: UDGraph<I>) -> Self {
        PhantomData
    }
}

/// Construct a HashMap of the type from a graph
///
/// The types should be connected according to the connections specified in the graph.
/// Afterwards, this HashMap can over multiple threads and used since all components
/// are connected according the the initial graph.
pub trait BuildFromGraph<I>
where
    Self: Sized,
    I: Clone + Eq + core::hash::Hash + Ord,
{
    /// Builds the HashMap
    fn build_from_graph(
        graph: UDGraph<I>,
    ) -> Result<std::collections::HashMap<I, Self>, IndexError>;
}

impl<I> BuildFromGraph<I> for PhantomData<I>
where
    Self: Sized,
    I: Clone + Eq + core::hash::Hash + Ord,
{
    fn build_from_graph(
        graph: UDGraph<I>,
    ) -> Result<std::collections::HashMap<I, Self>, IndexError> {
        Ok(graph
            .into_iter()
            .map(|(key, _)| (key, PhantomData::<I>))
            .collect())
    }
}

/* impl<I, T> From<UDGraph<I>> for Result<std::collections::HashMap<I, T>, IndexError>
where
    T: FromMap<I>,
    I: Ord + Clone + core::hash::Hash,
{
    fn from(value: UDGraph<I>) -> Self {
        let map = value.to_hash_map();
        T::from_map(&map)
    }
}*/

// TODO migrate to FromGraph eventually!
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
        I: Eq + core::hash::Hash + Clone + Ord;
}

impl<I> FromMap<I> for PhantomData<I> {
    fn from_map(map: &std::collections::HashMap<I, Vec<I>>) -> Result<HashMap<I, Self>, IndexError>
    where
        I: Eq + core::hash::Hash + Clone + Ord,
    {
        Ok(map
            .into_iter()
            .map(|(key, _)| (key.clone(), PhantomData::<I>))
            .collect())
    }
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
        I: Eq + core::hash::Hash + Clone + Ord,
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

impl<I> BuildFromGraph<I> for BarrierSync
where
    I: Clone + Eq + core::hash::Hash + std::cmp::Ord,
{
    fn build_from_graph(
        graph: UDGraph<I>,
    ) -> Result<std::collections::HashMap<I, Self>, IndexError> {
        let barrier = hurdles::Barrier::new(graph.len());
        let res = graph
            .nodes()
            .into_iter()
            .map(|key| {
                (
                    key.clone(),
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

pub struct NextTimePoint<F> {
    pub increment: F,
    pub time: F,
    pub iteration: i64,
    pub event: Option<TimeEvent>,
}

/// Increments time of the simulation
///
/// In the future we hope to add adaptive steppers depending on a specified accuracy function.
pub trait TimeStepper<F> {
    /// Advances the time stepper to the next time point. Also returns if there is an event
    /// scheduled to take place and the next time value and iteration number
    #[must_use]
    fn advance(&mut self) -> Result<Option<NextTimePoint<F>>, CalcError>;

    /// Retrieved the last point at which the simulation was fully recovered.
    /// This might be helpful in the future when error handling is more mature and able to recover.
    fn get_last_full_save(&self) -> Option<(F, i64)>;
}

/// Time stepping with a fixed time length
///
/// This time-stepper increments the time variable by the same length.
/// ```
/// # use cellular_raza_core::backend::chili::FixedStepsize;
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
    fn advance(&mut self) -> Result<Option<NextTimePoint<F>>, CalcError> {
        self.current_iteration += 1;
        self.current_time = F::from_i64(self.current_iteration).ok_or(CalcError(
            "Error when casting from i64 to floating point value".to_owned(),
        ))? * self.dt
            + self.t0;
        // TODO Check if a current event should take place
        let event = self
            .all_events
            .iter()
            .filter(|(_, iteration, _)| *iteration == self.current_iteration)
            .map(|(_, _, event)| event.clone())
            .last();

        if self.current_iteration <= self.maximum_iterations {
            Ok(Some(NextTimePoint {
                increment: self.dt,
                time: self.current_time,
                iteration: self.current_iteration,
                event,
            }))
        } else {
            Ok(None)
        }
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

/// Handles communications between different simulation processes.
///
/// Often times, information needs to be exchanged between threads.
/// For example, positional and force information of cells living at the boundary.
///
/// The receiver is referenced by the index `I` and will obtain the message `T`.
/// The trait was designed around the [crossbeam_channel] sender-receiver pair.
/// However, it should allow for more generic setups where eg. information could be shared
/// by different means such as sharing memory.
///
/// Between the [Communicator::send] and [Communicator::receive] method, a synchronization step
/// needs to happen. Otherwise, dataraces can occur and invalidate given results.
/// See the [Sync] trait for more details on syncing between threads.
pub trait Communicator<I, T>
where
    Self: Sized,
{
    /// Sends information to a particular receiver.
    fn send(&mut self, receiver: &I, message: T) -> Result<(), SimulationError>;
    /// Receives the information previously sent
    ///
    /// When implementing this trait, make sure to empty any existing queue or shared memory.
    /// Otherwise received messages will be stacking up, using up more memory+
    /// and yielding wrong results.
    fn receive(&mut self) -> Vec<T>;
}

/// Sender-Receiver [Communicator] based on [crossbeam_channel].
///
/// This struct contains one receiver and multiple senders.
/// It can be constructed by using the [FromMap] trait.
/// ```
/// # use cellular_raza_core::backend::chili::{ChannelComm, Communicator, FromMap};
/// # use std::collections::HashMap;
/// let map = HashMap::from([
///     (0, vec![1]),
///     (1, vec![0, 2]),
///     (2, vec![1]),
/// ]);
///
/// // Construct multiple communicators from a given map.
/// let mut channel_comms = ChannelComm::from_map(&map).unwrap();
///
/// // Send a message from 0 to 1
/// channel_comms.get_mut(&0).unwrap().send(&1, true);
/// channel_comms.get_mut(&0).unwrap().send(&1, false);
/// // Receive all elements at communicator 1
/// let elements = channel_comms.get_mut(&1).unwrap().receive();
///
/// assert_eq!(elements, vec![true, false]);
/// ```
#[derive(Clone)]
pub struct ChannelComm<I, T> {
    senders: std::collections::BTreeMap<I, crossbeam_channel::Sender<T>>,
    receiver: crossbeam_channel::Receiver<T>,
}

impl<T, I> FromMap<I> for ChannelComm<I, T>
where
    I: Ord,
{
    fn from_map(map: &HashMap<I, Vec<I>>) -> Result<HashMap<I, Self>, IndexError>
    where
        I: Clone + core::hash::Hash + Eq,
    {
        let channels: HashMap<_, _> = map
            .keys()
            .into_iter()
            .map(|sender_key| {
                let (s, r) = crossbeam_channel::unbounded::<T>();
                (sender_key, (s, r))
            })
            .collect();
        let mut comms = HashMap::new();
        for key in map.keys().into_iter() {
            let senders = map
                .get(&key)
                .ok_or(IndexError("Network of communicators could not be constructed due to incorrect entries in map".into()))?
                .clone()
                .into_iter()
                .map(|connected_key| (connected_key.clone(), channels[&connected_key].0.clone())).collect();

            let comm = ChannelComm {
                senders,
                receiver: channels[&key].1.clone(),
            };
            comms.insert(key.clone(), comm);
        }
        Ok(comms)
    }
}

#[cfg(test)]
mod test_channel_comm {
    use itertools::Itertools;

    use super::*;

    #[test]
    fn test_from_map() -> Result<(), IndexError> {
        let map = std::collections::HashMap::from([
            (0_usize, vec![3, 1]),
            (1_usize, vec![0, 2]),
            (2_usize, vec![1, 3]),
            (3_usize, vec![2, 0]),
        ]);
        assert!(validate_map(&map));
        let channel_comms = ChannelComm::<usize, ()>::from_map(&map)?;
        assert_eq!(channel_comms.len(), 4);
        for i in 0..4 {
            assert!(channel_comms.keys().contains(&i));
        }
        for i in 0..4 {
            let higher = (i + 1) % 4;
            let lower = (i + 5) % 4;
            assert!(channel_comms[&i].senders.contains_key(&lower));
            assert!(channel_comms[&i].senders.contains_key(&higher));
        }
        Ok(())
    }

    #[test]
    fn test_from_map_2() -> Result<(), Box<dyn std::error::Error>> {
        let map = std::collections::HashMap::from([
            (0, vec![1, 2, 3]),
            (1, vec![0, 2, 3]),
            (2, vec![0, 1, 3]),
            (3, vec![0, 1, 2]),
        ]);
        assert!(validate_map(&map));
        let channel_comms = ChannelComm::<usize, Option<()>>::from_map(&map)?;
        for i in 0..4 {
            assert!(channel_comms.keys().contains(&i));
        }
        for i in 0..4 {
            for j in 0..4 {
                let contains = channel_comms[&i].senders.keys().contains(&j);
                assert_eq!(contains, i != j);
            }
        }
        Ok(())
    }

    fn from_map_n_line(n: usize) -> Result<(), Box<dyn std::error::Error>> {
        if n <= 1 {
            return Ok(());
        }
        let mut map = std::collections::HashMap::from([(0, vec![1, n]), (n, vec![n - 1, 0])]);
        for i in 1..n {
            map.insert(i, vec![i - 1, i + 1]);
        }
        assert!(validate_map(&map));
        let mut channel_comms = ChannelComm::<usize, usize>::from_map(&map)?;
        channel_comms
            .iter_mut()
            .map(|(key, comm)| {
                let recv_key = (key + 1) % (n + 1);
                comm.send(&recv_key, *key)
            })
            .collect::<Result<Vec<_>, _>>()?;
        channel_comms.iter_mut().for_each(|(key, comm)| {
            let results = comm.receive();
            assert!(results.len() > 0);
            if key > &0 {
                assert_eq!(results, vec![key - 1]);
            } else {
                assert_eq!(results, vec![n]);
            }
        });
        Ok(())
    }

    #[test]
    fn from_map_lines() {
        for i in 2..100 {
            from_map_n_line(i).unwrap();
        }
    }

    #[test]
    fn test_send() -> Result<(), Box<dyn std::error::Error>> {
        let map = std::collections::HashMap::from([(1_usize, vec![2]), (2_usize, vec![1])]);
        let mut channel_comms = ChannelComm::<usize, bool>::from_map(&map)?;
        channel_comms.get_mut(&1).unwrap().send(&2, true)?;
        channel_comms.get_mut(&2).unwrap().send(&1, false)?;
        Ok(())
    }

    #[test]
    fn test_empty_receive() -> Result<(), Box<dyn std::error::Error>> {
        let map = std::collections::HashMap::from([(1_usize, vec![2]), (2_usize, vec![1])]);
        let mut channel_comms = ChannelComm::<usize, f64>::from_map(&map)?;
        for (_, comm) in channel_comms.iter_mut() {
            let received_elements = comm.receive().into_iter().collect::<Vec<_>>();
            assert_eq!(received_elements.len(), 0);
        }
        Ok(())
    }

    #[test]
    fn test_send_receive() -> Result<(), Box<dyn std::error::Error>> {
        let map =
            std::collections::HashMap::from([(0, vec![1, 2]), (1, vec![0, 2]), (2, vec![0, 1])]);
        let mut channel_comms = ChannelComm::from_map(&map)?;

        // Send a dummy value
        for (index, comm) in channel_comms.iter_mut() {
            let next_index = (index + 1) % map.len();
            comm.send(&next_index, next_index as f64)?;
        }

        // Receive the value
        for (index, comm) in channel_comms.iter_mut() {
            let received_elements = comm.receive().into_iter().collect::<Vec<_>>();
            assert_eq!(received_elements, vec![*index as f64]);
        }
        Ok(())
    }

    #[test]
    fn test_send_plain_voxel() -> Result<(), Box<dyn std::error::Error>> {
        use crate::backend::chili::SubDomainPlainIndex;
        let map = std::collections::HashMap::from([
            (
                SubDomainPlainIndex(0),
                vec![SubDomainPlainIndex(1), SubDomainPlainIndex(2)],
            ),
            (
                SubDomainPlainIndex(1),
                vec![SubDomainPlainIndex(0), SubDomainPlainIndex(2)],
            ),
            (
                SubDomainPlainIndex(2),
                vec![SubDomainPlainIndex(0), SubDomainPlainIndex(1)],
            ),
        ]);
        let mut channel_comms = ChannelComm::from_map(&map)?;

        // Send a dummy value
        for (index, comm) in channel_comms.iter_mut() {
            let index = index.0;
            let next_index = SubDomainPlainIndex((index + 1) % map.len());
            comm.send(&next_index, next_index)?;
        }

        // Receive the value
        for (index, comm) in channel_comms.iter_mut() {
            let received_elements = comm.receive().into_iter().collect::<Vec<_>>();
            assert_eq!(received_elements, vec![*index]);
        }

        Ok(())
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

/// Send about the position of cells between threads.
///
/// This type is used during the update steps for cellular mechanics
/// [update_mechanics_step_1](super::datastructures::SubDomainBox::update_mechanics_step_1).
/// The response to [PosInformation] is the [ForceInformation] type.
/// Upon requesting the acting force, by providing the information stored in this struct,
/// the requester obtains the needed information about acting forces.
/// See also the [cellular_raza_concepts::Interaction] trait.
pub struct PosInformation<Pos, Vel, Inf> {
    /// Current position
    pub pos: Pos,
    /// Current velocity
    pub vel: Vel,
    /// Information shared between cells
    pub info: Inf,
    /// Index of cell in stored vector
    ///
    /// When returning information, this property is needed in order
    /// to get the correct cell in the vector of cells and update its properties.
    pub cell_index_in_vector: usize,
    /// Voxel index of the sending cell.
    /// Information should be returned to this voxel.
    pub index_sender: VoxelPlainIndex,
    /// Voxel index of the voxel from which information is requested.
    /// This index is irrelevant after the initial query has been sent.
    pub index_receiver: VoxelPlainIndex,
}

/// Return type to the requested [PosInformation].
///
/// This type is returned after performing all necessary force calculations in
/// [update_mechanics_step_2](super::datastructures::SubDomainBox::update_mechanics_step_2).
/// The received information is then used in combination with the already present information
/// to update the position and velocity of cells in
/// [update_mechanics_step_3](super::datastructures::SubDomainBox::update_mechanics_step_3).
pub struct ForceInformation<For> {
    /// Overall force acting on cell.
    ///
    /// This force is already combined in the sense that multiple forces may be added together.
    pub force: For,
    /// Index of cell in stored vector
    ///
    /// This property works in tandem with [Self::index_sender] in order to send
    /// the calculated information to the correct cell and update its properties.
    pub cell_index_in_vector: usize,
    /// The voxel index where information is returned to
    pub index_sender: VoxelPlainIndex,
}

/// Send cell and its AuxStorage between threads.
pub struct SendCell<Cel, Aux>(pub Cel, pub Aux);

#[doc(hidden)]
#[allow(unused)]
mod test_derive_communicator {
    /// ```
    /// use cellular_raza_core::proc_macro::Communicator;
    /// use cellular_raza_core::backend::chili::{SimulationError, ChannelComm, Communicator};
    /// #[derive(Communicator)]
    /// #[CommunicatorCorePath(cellular_raza_core)]
    /// struct MyComm<I, T> {
    ///     #[Comm(I, T)]
    ///     comm: ChannelComm<I, T>
    /// }
    /// ```
    fn default() {}

    /// ```
    /// use cellular_raza_core::proc_macro::Communicator;
    /// use cellular_raza_core::backend::chili::{SimulationError, ChannelComm, Communicator};
    /// #[derive(Communicator)]
    /// #[CommunicatorCorePath(cellular_raza_core)]
    /// struct MyDouble<I> {
    ///     #[Comm(I, String)]
    ///     comm1: ChannelComm<I, String>,
    ///     #[Comm(I, f64)]
    ///     comm2: ChannelComm<I, f64>,
    /// }
    /// ```
    fn two_communicators_explicit() {}

    /// ```
    /// use cellular_raza_core::proc_macro::Communicator;
    /// use cellular_raza_core::backend::chili::{SimulationError, ChannelComm, Communicator};
    /// struct Message<T>(T);
    /// #[derive(Communicator)]
    /// #[CommunicatorCorePath(cellular_raza_core)]
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
mod test_derive_from_map {
    /// ```
    /// use cellular_raza_core::{
    ///     proc_macro::FromMap,
    ///     backend::chili::{ChannelComm, FromMap},
    /// };
    /// use cellular_raza_concepts::IndexError;
    /// #[derive(FromMap)]
    /// #[FromMapIndex(usize)]
    /// struct MyNewComm {
    ///     channel_comm_1: ChannelComm<usize, String>,
    ///     channel_comm_2: ChannelComm<usize, (f64, f32)>,
    /// }
    /// ```
    fn default() {}

    /// ```
    /// use cellular_raza_core::{
    ///     proc_macro::FromMap,
    ///     backend::chili::{ChannelComm, FromMap},
    /// };
    /// use cellular_raza_concepts::IndexError;
    /// #[derive(FromMap)]
    /// #[FromMapIndex(I)]
    /// struct MyNewComm<I> {
    ///     channel_comm_1: ChannelComm<I, String>,
    ///     channel_comm_2: ChannelComm<I, (f64, f32)>,
    /// }
    /// ```
    fn generic_index() {}

    /// ```
    /// use cellular_raza_core::{
    ///     proc_macro::FromMap,
    ///     backend::chili::{ChannelComm, FromMap},
    /// };
    /// use cellular_raza_concepts::IndexError;
    /// #[derive(FromMap)]
    /// #[FromMapIndex(i16)]
    /// struct MyNewComm<T>
    /// where
    ///     T: Clone,
    /// {
    ///     channel_comm_1: ChannelComm<i16, T>,
    ///     channel_comm_2: ChannelComm<i16, (f64, f32)>,
    /// }
    /// ```
    fn where_clause() {}

    /// ```
    /// use cellular_raza_core::{
    ///     proc_macro::FromMap,
    ///     backend::chili::{ChannelComm, FromMap},
    /// };
    /// use cellular_raza_concepts::IndexError;
    /// #[derive(FromMap)]
    /// #[FromMapIndex(I)]
    /// struct MyNewComm<I>
    /// where
    ///     I: std::fmt::Display,
    /// {
    ///     channel_comm_1: ChannelComm<I, f64>,
    ///     channel_comm_2: ChannelComm<I, (f64, f32)>,
    /// }
    /// ```
    fn where_clause_on_index() {}
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
            /// use cellular_raza_core::proc_macro::build_communicator;
            /// build_communicator!(
            ///     name: __MyComm,
            ///     aspects: [
            #[doc = stringify!($($asp),*)]
            ///     ],
            ///     core_path: cellular_raza_core
            /// );
            /// ```
            #[allow(non_snake_case)]
            fn $func_name () {}
        };
    );

    cellular_raza_core_proc_macro::run_test_for_aspects!(
        test: test_build_communicator,
        aspects: [Mechanics, Interaction, Cycle, Reactions]
    );

    /// ```compile_fail
    /// build_communicator!(
    ///     name: __MyComm,
    ///     aspects: [Mechanics, Cycle],
    /// );
    /// ```
    fn without_path() {}
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
        assert_eq!(t0, time_stepper.current_time);
        assert_eq!(0.2, time_stepper.dt);
        assert_eq!(0, time_stepper.current_iteration);
        assert_eq!(None, time_stepper.current_event);
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

        for i in 1..11 {
            let next = time_stepper.advance().unwrap().unwrap();
            assert_eq!(dt, next.increment);
            assert_eq!(t0 + i as f64 * dt, next.time);
            assert_eq!(i as i64, next.iteration);
            if i == 10 {
                assert_eq!(Some(TimeEvent::PartialSave), next.event);
            } else {
                assert_eq!(None, next.event);
            }
        }
    }

    #[test]
    fn stepping_2() {
        let t0 = 0.0;
        let dt = 0.1;
        let save_points = vec![0.5, 0.7, 0.9, 1.0];
        let mut time_stepper =
            FixedStepsize::from_save_points(t0, dt, save_points.clone()).unwrap();

        assert_eq!(t0, time_stepper.current_time);
        for i in 1..11 {
            let next = time_stepper.advance().unwrap().unwrap();
            assert_eq!(dt, next.increment);
            assert_eq!(t0 + i as f64 * dt, next.time);
            assert_eq!(i as i64, next.iteration);
            if save_points.contains(&next.time) {
                assert_eq!(Some(TimeEvent::PartialSave), next.event);
            }
        }
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
