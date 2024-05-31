use crate::storage::StorageError;
use cellular_raza_concepts::*;
use core::any::type_name;
use core::fmt::{Debug, Display};

use std::error::Error; // TODO in the future use core::error::Error (unstable right now)

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
///
/// This main error type should be derivable from errors arising during the simulation process.
/// It is required for custom error types `MyCustomError` of the engine to implement the
/// `From<MyCustomError> for SimulationError`.
///
/// Errors are separated by their ability to be recovered, ignored or handled otherwise.
/// Since this crate aims to provide an adaptive solving approach, it is desired to have a fallback
/// mechanism which can be called for errors which may arise due to precision problems.
#[derive(Debug)]
pub enum SimulationError {
    // Very likely to be user errors
    /// Occurs during calculations of any mathematical update steps such as
    /// [Interaction](cellular_raza_concepts::Interaction) between cells.
    CalcError(CalcError),
    /// Related to time-stepping events. See [crate::time].
    TimeError(TimeError),
    /// Error-type specifically related to the [Controller](cellular_raza_concepts::Controller)
    /// trait.
    ControllerError(ControllerError),
    /// An error specific to cell-division events by the
    ///
    /// [Cycle](cellular_raza_concepts::Cycle) trait.
    DivisionError(DivisionError),
    /// Related to the [PhasedDeath](cellular_raza_concepts::CycleEvent::PhasedDeath) event.
    /// This error can only occur during the
    /// [update_conditional_phased_death](cellular_raza_concepts::Cycle::update_conditional_phased_death)
    /// method.
    DeathError(DeathError),
    /// Enforcing boundary conditions on cells can exhibhit this boundary error.
    BoundaryError(BoundaryError),
    /// Plotting results. See also [cellular_raza_concepts::PlotSelf] and [cellular_raza_concepts::CreatePlottingRoot].
    DrawingError(DrawingError),
    /// Mostly caused by trying to find a voxel by its index.
    /// This error can also occur when applying too large simulation-steps.
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
    /// Occurs when requested data could not be returned
    RequestError(RequestError),
    /// Errors related to [rand] and its functionalities
    RngError(RngError),

    // Highly unlikely to be user errors
    /// Errors surrounding construction of [rayon::ThreadPool].
    ThreadingError(rayon::ThreadPoolBuildError),
}

impl_from_error! {SimulationError,
    (ReceiveError, RecvError),
    (CalcError, CalcError),
    (ControllerError, ControllerError),
    (RequestError, RequestError),
    (TimeError, TimeError),
    (DivisionError, DivisionError),
    (DeathError, DeathError),
    (BoundaryError, BoundaryError),
    (IndexError, IndexError),
    (IoError, std::io::Error),
    (DrawingError, DrawingError),
    (RngError, RngError),
    (StorageError, StorageError),
    (ThreadingError, rayon::ThreadPoolBuildError)
}

impl_error_variant! {SimulationError,
    SendError,
    ReceiveError,
    RequestError,
    CalcError,
    ControllerError,
    TimeError,
    DivisionError,
    DeathError,
    BoundaryError,
    IndexError,
    IoError,
    DrawingError,
    RngError,
    StorageError,
    ThreadingError
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
