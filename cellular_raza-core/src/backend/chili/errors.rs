pub use crate::storage::StorageError;
use cellular_raza_concepts::*;
pub use cellular_raza_concepts::{DecomposeError, IndexError};
use core::any::type_name;
use core::fmt::{Debug, Display};

use std::error::Error;
use std::sync::atomic::AtomicBool;
use std::sync::Arc; // TODO in the future use core::error::Error (unstable right now)

use crossbeam_channel::{RecvError, SendError};

macro_rules! impl_error_variant {
    ($name: ident, $($err_var: ident),+) => {
        // Implement Display for ErrorVariant
        impl Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(
                        $name::$err_var(message) => write!(f, "{}", message),
                    )+
                }
            }
        }
    }
}

macro_rules! impl_from_error {
    ($name: ident, $(($err_var: ident, $err_type: ty)),+) => {
        $(
            // Implement conversion from error to errorvariant
            impl From<$err_type> for $name {
                fn from(err: $err_type) -> Self {
                    $name::$err_var(err)
                }
            }
        )+
    }
}

/// Covers all errors that can occur in this Simulation
/// The errors are listed from very likely to be a user error from almost certainly an internal error.
///
/// # Categorization of errors
/// Some errors are more likely to be occurring due to an incorrect usage by an end user
/// while others are highly likely to be due to some internal implementation problem.
/// Independent of the exact reason, why they are occurring, some can be handled explicitly
/// while others force an abort of the simulation. See also [HandlingStrategy].
///
/// | Variant | Concept Implementation | Engine-Fault | Time Increment |
/// | --- |:---:|:---:|:---:|
/// | [CalcError](SimulationError::CalcError) | 8/10 | 2/10 | 0/10 |
/// | [DecomposeError](SimulationError::DecomposeError) | 7/10 | 3/10 | 0/10 |
/// | [ControllerError](SimulationError::ControllerError) | 8/10 | 1/10 | 1/10 |
/// | [DivisionError](SimulationError::DivisionError) | 8/10 | 1/10 | 1/10 |
/// | [DeathError](SimulationError::DeathError) | 8/10 | 1/10 | 1/10 |
/// | [BoundaryError](SimulationError::BoundaryError) | 7/10 | 1/10 | 2/10 |
/// | [DrawingError](SimulationError::DrawingError) | 9/10 | 1/10 | 0/10 |
/// | [IndexError](SimulationError::IndexError) | 6/10 | 2/10 | 2/10 |
/// | [SendError](SimulationError::SendError) | 3/10 | 6/10 | 1/10 |
/// | [ReceiveError](SimulationError::ReceiveError) | 3/10 | 6/10 | 1/10 |
/// | [StorageError](SimulationError::StorageError) | 3/10 | 7/10 | 0/10 |
/// | [RngError](SimulationError::RngError) | 1/10 | 8/10 | 1/10 |
/// | [IoError](SimulationError::IoError) | 1/10 | 9/10 | 0/10 |
#[derive(Debug)]
pub enum SimulationError {
    // Very likely to be user errors
    /// Occurs during calculations of any mathematical update steps such as
    /// [Interaction](cellular_raza_concepts::Interaction) between cells.
    CalcError(CalcError),
    /// Error associated to splitting up the simulation domain into chunks
    /// in order to effectively process it in parallel.
    DecomposeError(DecomposeError),
    /// Related to time-stepping events. See [crate::time].
    TimeError(TimeError),
    /// Error-type specifically related to the
    /// [Controller](cellular_raza_concepts::domain_old::Controller) trait.
    ControllerError(ControllerError),
    /// An error specific to cell-division events by the
    ///
    /// [Cycle](cellular_raza_concepts::Cycle) trait.
    DivisionError(DivisionError),
    /// Related to the [PhasedDeath](cellular_raza_concepts::CycleEvent::PhasedDeath) event.
    /// This error can only occcurr during the
    /// [update_conditional_phased_death](cellular_raza_concepts::Cycle::update_conditional_phased_death)
    /// method.
    DeathError(DeathError),
    /// Enforcing boundary conditions on cells can exhibhit this boundary error.
    BoundaryError(BoundaryError),
    /// Plotting results. See also [cellular_raza_concepts::PlotSelf]
    /// and [cellular_raza_concepts::CreatePlottingRoot].
    DrawingError(DrawingError),
    /// Mostly caused by trying to find a voxel by its index.
    /// This error can also occcurr when applying too large simulation-steps.
    IndexError(IndexError),

    // Less likely but possible to be user errors
    /// Sending information between threads fails
    SendError(String),
    /// Receiving information from another thread fails
    ReceiveError(RecvError),
    /// Storing results fails
    StorageError(StorageError),

    // Highly unlikely to be user errors
    /// When writing to output files or reading from them.
    /// See [std::io::Error]
    IoError(std::io::Error),

    /// Error related to random number generation by the [rand_chacha::ChaCha8Rng] struct.
    RngError(RngError),

    /// Only occurs when another thread returns an error
    OtherThreadError(String),
}

impl_from_error! {SimulationError,
    (ReceiveError, RecvError),
    (CalcError, CalcError),
    (DecomposeError, DecomposeError),
    (TimeError, TimeError),
    (ControllerError, ControllerError),
    (DivisionError, DivisionError),
    (DeathError, DeathError),
    (BoundaryError, BoundaryError),
    (IndexError, IndexError),
    (IoError, std::io::Error),
    (DrawingError, DrawingError),
    (StorageError, StorageError),
    (RngError, RngError)
}

impl_error_variant! {SimulationError,
    SendError,
    ReceiveError,
    CalcError,
    DecomposeError,
    TimeError,
    ControllerError,
    DivisionError,
    DeathError,
    BoundaryError,
    IndexError,
    IoError,
    DrawingError,
    StorageError,
    RngError,
    OtherThreadError
}

// Implement the general error property
impl std::error::Error for SimulationError {}

// Implement conversion from Sending error manually
impl<T> From<SendError<T>> for SimulationError {
    fn from(_err: SendError<T>) -> Self {
        SimulationError::SendError(format!(
            "Error receiving object of type {}",
            type_name::<SendError<T>>()
        ))
    }
}

impl<E> From<plotters::drawing::DrawingAreaErrorKind<E>> for SimulationError
where
    E: Error + Send + Sync,
{
    fn from(drawing_error: plotters::drawing::DrawingAreaErrorKind<E>) -> SimulationError {
        SimulationError::DrawingError(DrawingError::from(drawing_error))
    }
}

/// Contains handling strategies for errors which can arise during the simulation process.
///
/// # Handling Strategies
/// A handler has multiple options on how to approach a recovery in a simulation.
///
/// [RevertChangeAccuracy](HandlingStrategy::RevertChangeAccuracy)
///
/// One option is to revert to the last known full snapshot of the simulation, increase the
/// accuracy of the solvers and try again from there. This requires that a working, deterministic
/// and accurate serialization/deserialization of the whole simulation state is setup (see
/// [storage](crate::storage) module for more details).
// TODO implement more of this
pub enum HandlingStrategy {
    /// Throw away current results, return to last known good simulation state.
    /// Load these results (possibly from disk) and try again with increased accuracy.
    RevertChangeAccuracy,
    /// Throw away current results and stop the simulation entire.
    /// This represents a non-recoverable state.
    Abort,
    /// Ignores the error and continues with the simulation
    Ignore,
}

/// Handle any error which may occur as specified by [SimulationError]
pub trait ErrorHandler: Sized {
    /// Creates multiple handlers which are responsible for individual threads.
    fn spawn_multiple(n_threads: usize) -> impl IntoIterator<Item = Self>;

    /// Accumulates an error event.
    fn handle_event(&mut self, value: Result<(), SimulationError>);

    /// Determines possible [HandlingStrategy] after having accumulated all possible errors.
    fn determine_strategy(&self, error: SimulationError) -> HandlingStrategy {
        use HandlingStrategy::*;
        use SimulationError::*;
        // TODO make error handling more nuanced to be valuable
        match error {
            SendError(_) => Abort,
            ReceiveError(_) => RevertChangeAccuracy,
            CalcError(_) => RevertChangeAccuracy,
            DecomposeError(_) => Abort,
            TimeError(_) => Abort,
            ControllerError(_) => Abort,
            DivisionError(_) => RevertChangeAccuracy,
            DeathError(_) => RevertChangeAccuracy,
            BoundaryError(_) => Abort,
            IndexError(_) => RevertChangeAccuracy,
            IoError(_) => Ignore,
            DrawingError(_) => Ignore,
            StorageError(_) => Ignore,
            RngError(_) => Abort,
            OtherThreadError(_) => Abort,
        }
    }

    /// Determines if the simulation is able to be continued or not.
    fn do_continue(&mut self) -> Result<(), SimulationError>;
}

/// TODO
#[allow(unused)]
pub struct DefaultHandler {
    thread: usize,
    errors: Vec<SimulationError>,
    do_continue: Arc<AtomicBool>,
}

impl ErrorHandler for DefaultHandler {
    fn spawn_multiple(n_threads: usize) -> impl IntoIterator<Item = Self> {
        let do_continue = Arc::new(AtomicBool::new(true));
        (0..n_threads).map(move |thread| Self {
            thread,
            errors: vec![],
            do_continue: Arc::clone(&do_continue),
        })
    }
    fn handle_event(&mut self, error: Result<(), SimulationError>) {
        match error {
            Ok(_) => (),
            Err(e) => self.errors.push(e),
        }
    }

    fn do_continue(&mut self) -> Result<(), SimulationError> {
        self.errors
            .iter()
            .for_each(|e| println!("thread {:2} detected error: {}", self.thread, e));
        match self.errors.drain(..).next() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(feature = "pyo3")]
impl From<SimulationError> for pyo3::PyErr {
    fn from(value: SimulationError) -> Self {
        use pyo3::exceptions::*;
        use SimulationError::*;
        match value {
            CalcError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            DecomposeError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            TimeError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            ControllerError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            DivisionError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            DeathError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            BoundaryError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            DrawingError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            IndexError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            SendError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            ReceiveError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            StorageError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            RngError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
            IoError(e) => pyo3::PyErr::new::<PyIOError, _>(format!("cr_err: {e:?}")),
            OtherThreadError(e) => pyo3::PyErr::new::<PyValueError, _>(format!("cr_err: {e:?}")),
        }
    }
}
