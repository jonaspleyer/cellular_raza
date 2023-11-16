pub use crate::storage::concepts::StorageError;
use cellular_raza_concepts::errors::*;
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
/// The errors are listed from very likely to be a user error from almost certainly an internal error.
#[derive(Debug)]
pub enum SimulationError {
    // Very likely to be user errors
    CalcError(CalcError),
    ControllerError(ControllerError),
    RequestError(RequestError),
    StepsizeError(StepsizeError),
    DivisionError(DivisionError),
    DeathError(DeathError),
    BoundaryError(BoundaryError),
    DrawingError(DrawingError),

    // Less likely but possible to be user errors
    SendError(String),
    ReceiveError(RecvError),
    StorageError(StorageError),

    // Highly unlikely to be user errors
    IndexError(IndexError),
    IoError(std::io::Error),
    ThreadingError(rayon::ThreadPoolBuildError),
}

impl_from_error! {SimulationError,
    (ReceiveError, RecvError),
    (CalcError, CalcError),
    (ControllerError, ControllerError),
    (RequestError, RequestError),
    (StepsizeError, StepsizeError),
    (DivisionError, DivisionError),
    (DeathError, DeathError),
    (BoundaryError, BoundaryError),
    (IndexError, IndexError),
    (IoError, std::io::Error),
    (DrawingError, DrawingError),
    (StorageError, StorageError),
    (ThreadingError, rayon::ThreadPoolBuildError)
}

impl_error_variant! {SimulationError,
    SendError,
    ReceiveError,
    RequestError,
    CalcError,
    ControllerError,
    StepsizeError,
    DivisionError,
    DeathError,
    BoundaryError,
    IndexError,
    IoError,
    DrawingError,
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

/// Contains handling strategies for errors which can arise during the simulation process.
///
/// # Handling Strategies
/// A handler has multiple options on how to approach a recovery in a simulation.
///
/// [RevertIncreaseAccuracy](HandlingStrategies::RevertChangeAccuracy)
///
/// One option is to revert to the last known full snapshot of the simulation, increase the
/// accuracy of the solvers and try again from there. This requires that a working, deterministic
/// and accurate serialization/deserialization of the whole simulation state is setup (see
/// [storage](crate::storage) module for more details).
// TODO implement more of this
pub enum HandlingStrategies {
    RevertChangeAccuracy,
    AbortSimulation,
}
