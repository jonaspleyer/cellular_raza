use std::collections::HashMap;


///
/// This very simple implementation uses the [Barrier](hurdles::Barrier) struct from the [hurdles] crate
/// which should in theory perform faster than the [std::sync::Barrier] struct from the standard library.
///
/// By using the [SyncSubDomains] trait, we can automatically create a collection of syncers
/// which can then be simply given to the respective threads and handle synchronization.
/// ```
/// # use std::collections::HashMap;
/// # use cellular_raza_core::backend::chili::simulation_flow::{BarrierSync, SyncSubDomains};
/// let map = HashMap::from_iter([
///     (0, vec![1]),
///     (1, vec![0]),
/// ]);
/// let mut syncers = BarrierSync::from_map(map);
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

/// Responsible for syncing the simulation between different threads.
pub trait SyncSubDomains
where
    Self: Sized,
{
    /// [SubDomains](super::concepts::SubDomain) can be neighboring each other via complicated graphs.
    /// An easy way to represent this is by using a [HashMap]. We want to create Barriers which match
    /// the specified subdomain indices (given in [usize]).
    fn from_map(map: std::collections::HashMap<usize, Vec<usize>>) -> HashMap<usize, Self>;

    /// Function which forces connected syncers to wait for each other.
    /// This approach does not necessarily require all threads to wait but can mean that
    /// only depending threads wait for each other.
    fn sync(&mut self);
}

impl SyncSubDomains for BarrierSync {
    fn from_map(map: std::collections::HashMap<usize, Vec<usize>>) -> HashMap<usize, Self> {
        let barrier = hurdles::Barrier::new(map.len());
        map.keys()
            .map(|&i| {
                (
                    i,
                    Self {
                        barrier: barrier.clone(),
                    },
                )
            })
            .collect()
    }

    fn sync(&mut self) {
        self.barrier.wait();
    }
}

#[cfg(test)]
pub mod test_sync {
    use super::*;
    use std::sync::*;

    fn test_single_map<S>(map: HashMap<usize, Vec<usize>>)
    where
        S: 'static + SyncSubDomains + Send + Sync,
    {
        // Define the number of threads and iterations to use
        let n_iterations = 1_000;
        let n_threads = map.len();

        // We count the number of iterations via this mutex.
        // Individual threads will increment their counter by +1 each time they are executed
        let iteration_counter = Arc::new(
            Mutex::new(
                Vec::from_iter((0..n_threads).map(|_| 0_usize))
            )
        );

        // Create a barrier from which we
        let syncers = S::from_map(map);
        let handles = syncers.into_iter()
            .map(|(n_thread, mut syncer)| {
                let iteration_counter_thread = Arc::clone(&iteration_counter);
                std::thread::spawn(move || {
                    for n_iteration in 0..n_iterations {
                        syncer.sync();
                        iteration_counter_thread.lock().unwrap()[n_thread] += 1;
                        syncer.sync();
                        let current_value = iteration_counter_thread.lock().unwrap().clone();
                        assert_eq!(
                            current_value,
                            vec![n_iteration + 1; n_threads]
                        );
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
        S: 'static + SyncSubDomains + Send + Sync,
    {
        let map0 = HashMap::from_iter([
            (0, vec![1]),
            (1, vec![0]),
        ]);
        test_single_map::<S>(map0);

        let map1 = HashMap::from_iter([
            (0, vec![1,2]),
            (1, vec![0,2]),
            (2, vec![0,1]),
        ]);
        test_single_map::<S>(map1);

        let map2 = HashMap::from_iter([
            (0, vec![1,2,3]),
            (1, vec![0,2,3]),
            (2, vec![0,1,3]),
            (3, vec![0,1,2]),
        ]);
        test_single_map::<S>(map2);

        let map3 = HashMap::from_iter([
            (0, vec![1,2]),
            (1, vec![0,3]),
            (2, vec![0,3]),
            (3, vec![1,2]),
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
