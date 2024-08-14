use cellular_raza_concepts::IndexError;
use itertools::Itertools;

use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

use super::errors::SimulationError;

///
/// This very simple implementation uses the [hurdles::Barrier] struct which should
/// in theory perform faster than the [std::sync::Barrier] struct from the standard library.
///
/// By using the [SyncSubDomains] trait, we can automatically create a collection of syncers
/// which can then be simply given to the respective threads and handle synchronization.
/// ```
/// # use std::collections::BTreeMap;
/// # use cellular_raza_core::backend::chili::{BarrierSync, FromMap, SyncSubDomains};
/// let map = BTreeMap::from_iter([
///     (0, std::collections::BTreeSet::from([1])),
///     (1, std::collections::BTreeSet::from([0])),
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
/// This algorithm checks if every keys neighbors also contain the specified key.
/// If this is not the case, the map cannot be considered valid.
/// Note that this algorithm does not check if all keys are connected.
/// This means, disjoint sets are allowed.
///
/// ```
/// use cellular_raza_core::backend::chili::validate_map;
///
/// let new_map = std::collections::BTreeMap::from([
///     (1_usize, std::collections::BTreeSet::from([0,2])),
///     (2_usize, std::collections::BTreeSet::from([1,3])),
///     (3_usize, std::collections::BTreeSet::from([2,0])),
///     (0_usize, std::collections::BTreeSet::from([3,1])),
/// ]);
///
/// let is_valid = validate_map(&new_map);
/// assert_eq!(is_valid, true);
/// ```
pub fn validate_map<I>(map: &std::collections::BTreeMap<I, BTreeSet<I>>) -> bool
where
    I: Eq + core::hash::Hash + Clone + Ord,
{
    for (index, neighbors) in map.iter() {
        if neighbors.iter().any(|n| match map.get(n) {
            Some(reverse_neighbors) => !reverse_neighbors.contains(index),
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
    I: core::hash::Hash + Clone + Eq + Ord,
{
    /// Convert the [UDGraph] into a regular [BTreeMap].
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
    pub fn to_btree(self) -> BTreeMap<I, BTreeSet<I>> {
        let mut map: BTreeMap<_, BTreeSet<I>> = self
            .0
            .iter()
            .map(|x| {
                [
                    (x.0.clone(), BTreeSet::new()),
                    (x.1.clone(), BTreeSet::new()),
                ]
                .into_iter()
            })
            .flatten()
            .collect();
        self.0.iter().for_each(|(c1, c2)| {
            map.entry(c1.clone()).and_modify(|v| {
                v.insert(c2.clone());
            });
            map.entry(c2.clone()).and_modify(|v| {
                v.insert(c1.clone());
            });
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

/// Construct a BTreeMap of the type from a graph
///
/// The types should be connected according to the connections specified in the graph.
/// Afterwards, this BTreeMap can over multiple threads and used since all components
/// are connected according the the initial graph.
pub trait BuildFromGraph<I>
where
    Self: Sized,
    I: Clone + Eq + core::hash::Hash + Ord,
{
    /// Builds the BTreeMap
    fn build_from_graph(graph: UDGraph<I>) -> Result<BTreeMap<I, Self>, IndexError>;
}

impl<I> BuildFromGraph<I> for PhantomData<I>
where
    Self: Sized,
    I: Clone + Eq + core::hash::Hash + Ord,
{
    fn build_from_graph(graph: UDGraph<I>) -> Result<BTreeMap<I, Self>, IndexError> {
        Ok(graph
            .into_iter()
            .map(|(key, _)| (key, PhantomData::<I>))
            .collect())
    }
}

// TODO migrate to FromGraph eventually!
/// Constructs a collection of Items from a map (graph)
pub trait FromMap<I>
where
    Self: Sized,
{
    /// [SubDomains](cellular_raza_concepts::SubDomain) can be neighboring each
    /// other via complicated graphs.
    /// An easy way to represent this is by using a [BTreeMap]. We want to create Barriers which match
    /// the specified subdomain indices.
    fn from_map(map: &BTreeMap<I, BTreeSet<I>>) -> Result<BTreeMap<I, Self>, IndexError>
    where
        I: Eq + core::hash::Hash + Clone + Ord;
}

impl<I> FromMap<I> for PhantomData<I> {
    fn from_map(map: &BTreeMap<I, BTreeSet<I>>) -> Result<BTreeMap<I, Self>, IndexError>
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
    fn from_map(map: &BTreeMap<I, BTreeSet<I>>) -> Result<BTreeMap<I, Self>, IndexError>
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
    fn build_from_graph(graph: UDGraph<I>) -> Result<BTreeMap<I, Self>, IndexError> {
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
/// let map = std::collections::BTreeMap::from([
///     (0, std::collections::BTreeSet::from([1])),
///     (1, std::collections::BTreeSet::from([0, 2])),
///     (2, std::collections::BTreeSet::from([1])),
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
    fn from_map(map: &BTreeMap<I, BTreeSet<I>>) -> Result<BTreeMap<I, Self>, IndexError>
    where
        I: Clone + core::hash::Hash + Eq,
    {
        let channels: BTreeMap<_, _> = map
            .keys()
            .into_iter()
            .map(|sender_key| {
                let (s, r) = crossbeam_channel::unbounded::<T>();
                (sender_key, (s, r))
            })
            .collect();
        let mut comms = BTreeMap::new();
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
        let map = BTreeMap::from([
            (0_usize, BTreeSet::from([3, 1])),
            (1_usize, BTreeSet::from([0, 2])),
            (2_usize, BTreeSet::from([1, 3])),
            (3_usize, BTreeSet::from([2, 0])),
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
        let map = BTreeMap::from([
            (0, BTreeSet::from([1, 2, 3])),
            (1, BTreeSet::from([0, 2, 3])),
            (2, BTreeSet::from([0, 1, 3])),
            (3, BTreeSet::from([0, 1, 2])),
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
        let mut map =
            BTreeMap::from([(0, BTreeSet::from([1, n])), (n, BTreeSet::from([n - 1, 0]))]);
        for i in 1..n {
            map.insert(i, BTreeSet::from([i - 1, i + 1]));
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
        let map = BTreeMap::from([
            (1_usize, BTreeSet::from([2])),
            (2_usize, BTreeSet::from([1])),
        ]);
        let mut channel_comms = ChannelComm::<usize, bool>::from_map(&map)?;
        channel_comms.get_mut(&1).unwrap().send(&2, true)?;
        channel_comms.get_mut(&2).unwrap().send(&1, false)?;
        Ok(())
    }

    #[test]
    fn test_empty_receive() -> Result<(), Box<dyn std::error::Error>> {
        let map = BTreeMap::from([
            (1_usize, BTreeSet::from([2])),
            (2_usize, BTreeSet::from([1])),
        ]);
        let mut channel_comms = ChannelComm::<usize, f64>::from_map(&map)?;
        for (_, comm) in channel_comms.iter_mut() {
            let received_elements = comm.receive().into_iter().collect::<Vec<_>>();
            assert_eq!(received_elements.len(), 0);
        }
        Ok(())
    }

    #[test]
    fn test_send_receive() -> Result<(), Box<dyn std::error::Error>> {
        let map = BTreeMap::from([
            (0, BTreeSet::from([1, 2])),
            (1, BTreeSet::from([0, 2])),
            (2, BTreeSet::from([0, 1])),
        ]);
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
        let map = BTreeMap::from([
            (
                SubDomainPlainIndex(0),
                BTreeSet::from([SubDomainPlainIndex(1), SubDomainPlainIndex(2)]),
            ),
            (
                SubDomainPlainIndex(1),
                BTreeSet::from([SubDomainPlainIndex(0), SubDomainPlainIndex(2)]),
            ),
            (
                SubDomainPlainIndex(2),
                BTreeSet::from([SubDomainPlainIndex(0), SubDomainPlainIndex(1)]),
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
        let sender = self
            .senders
            .get(&receiver)
            .ok_or(super::IndexError(format!(
                "could not find specified receiver"
            )))?;
        sender.send(message)?;
        Ok(())
    }
}

#[doc(hidden)]
#[allow(unused)]
mod test_derive_communicator {
    /// ```
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
            /// use cellular_raza_core::backend::chili::build_communicator;
            /// build_communicator!(
            ///     name: __MyComm,
            ///     aspects: [
            #[doc = stringify!($($asp),*)]
            ///     ],
            ///     core_path: cellular_raza_core
            /// );
            /// let mut map = std::collections::BTreeMap::new();
            /// map.insert(0, std::collections::BTreeSet::from([1]));
            /// map.insert(1, std::collections::BTreeSet::from([0]));
            /// use cellular_raza_core::backend::chili::{ReactionsContactInformation, FromMap,
            /// Communicator, PosInformation, ForceInformation, VoxelPlainIndex, SendCell};
            /// let mut communicator = __MyComm::from_map(&map).unwrap().remove(&0).unwrap();
            /// macro_rules! test_aspect (
            ///     (Mechanics) => {
            ///         communicator.send(&1, SendCell(
            ///             VoxelPlainIndex::new(1),
            ///             format!("MyCell"),
            ///             format!("AuxStorage")
            ///         ));
            ///     };
            ///     (iInteraction) => {
            #[doc = concat!($(
                concat!("test_aspect!(", stringify!($asp), ", Interaction);"),
            )*)]
            ///     };
            ///     (Interaction) => {
            ///         communicator.send(&1, PosInformation {
            ///             pos: 1u8,
            ///             vel: 1.0,
            ///             info: (),
            ///             cell_index_in_vector: 1,
            ///             index_sender: VoxelPlainIndex::new(0),
            ///             index_receiver: VoxelPlainIndex::new(1),
            ///         });
            ///         communicator.send(&1, ForceInformation {
            ///             force: 0.1,
            ///             cell_index_in_vector: 0,
            ///             index_sender: VoxelPlainIndex::new(0),
            ///         });
            ///     };
            ///     ($asp:ident, Interaction) => {};
            ///     (Cycle) => {};
            ///     (Reactions) => {};
            ///     (ReactionsContact) => {
            ///         communicator.send(&1, ReactionsContactInformation {
            ///             pos: 1u8,
            ///             intracellular: [0.0, 1.0],
            ///             info: "hi",
            ///             cell_index_in_vector: 0,
            ///             index_sender: VoxelPlainIndex::new(0),
            ///             index_receiver: VoxelPlainIndex::new(1),
            ///         });
            ///     };
            /// );
            #[doc = concat!($(
                concat!("test_aspect!(", stringify!($asp), ");")
            ,)*)]
            /// ```
            #[allow(non_snake_case)]
            fn $func_name () {}
        };
    );

    cellular_raza_core_proc_macro::run_test_for_aspects!(
        test: test_build_communicator,
        aspects: [Mechanics, Interaction, Cycle, Reactions, ReactionsContact]
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

    fn test_single_map<S>(map: BTreeMap<usize, BTreeSet<usize>>)
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
        let map0 = BTreeMap::from_iter([(0, BTreeSet::from([1])), (1, BTreeSet::from([0]))]);
        test_single_map::<S>(map0);

        let map1 = BTreeMap::from_iter([
            (0, BTreeSet::from([1, 2])),
            (1, BTreeSet::from([0, 2])),
            (2, BTreeSet::from([0, 1])),
        ]);
        test_single_map::<S>(map1);

        let map2 = BTreeMap::from_iter([
            (0, BTreeSet::from([1, 2, 3])),
            (1, BTreeSet::from([0, 2, 3])),
            (2, BTreeSet::from([0, 1, 3])),
            (3, BTreeSet::from([0, 1, 2])),
        ]);
        test_single_map::<S>(map2);

        let map3 = BTreeMap::from_iter([
            (0, BTreeSet::from([1, 2])),
            (1, BTreeSet::from([0, 3])),
            (2, BTreeSet::from([0, 3])),
            (3, BTreeSet::from([1, 2])),
        ]);
        test_single_map::<S>(map3);

        let map4 = BTreeMap::from_iter([
            (0, BTreeSet::from([1])),
            (1, BTreeSet::from([2])),
            (2, BTreeSet::from([3])),
            (3, BTreeSet::from([4])),
            (4, BTreeSet::from([0])),
        ]);
        test_single_map::<BarrierSync>(map4);
    }

    #[test]
    fn barrier_sync() {
        test_multiple_maps::<BarrierSync>();
    }
}
