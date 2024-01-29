pub use crate::storage::StorageError;
pub use cellular_raza_concepts::IndexError;
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
/// The errors are listed from very likely to be a user error from almost certainly an internal error.
///
/// # Categorization of errors
/// Some errors are more likely to be occuring due to an incorrect usage by an end user
/// while others are highly likely to be due to some internal implementation problem.
/// Independent of the exact reason, why they are occuring, some can be handled explicitly
/// while others force an abort of the simulation. See also [HandlingStrategies].
///
/// | Variant | Concept Implementation | Engine-Fault | Time Increment |
/// | --- |:---:|:---:|:---:|
/// | [CalcError](SimulationError::CalcError) | 8/10 | 2/10 | 0/10 |
/// | [ControllerError](SimulationError::ControllerError) | 8/10 | 1/10 | 1/10 |
/// | [DivisionError](SimulationError::DivisionError) | 8/10 | 1/10 | 1/10 |
/// | [DeathError](SimulationError::DeathError) | 8/10 | 1/10 | 1/10 |
/// | [BoundaryError](SimulationError::BoundaryError) | 7/10 | 1/10 | 2/10 |
/// | [DrawingError](SimulationError::DrawingError) | 9/10 | 1/10 | 0/10 |
/// | [IndexError](SimulationError::IndexError) | 6/10 | 2/10 | 2/10 |
/// | [SendError](SimulationError::SendError) | 3/10 | 6/10 | 1/10 |
/// | [ReceiveError](SimulationError::ReceiveError) | 3/10 | 6/10 | 1/10 |
/// | [StorageError](SimulationError::StorageError) | 3/10 | 7/10 | 0/10 |
/// | [IoError](SimulationError::IoError) | 1/10 | 9/10 | 0/10 |
/// | [ThreadingError](SimulationError::ThreadingError) | 1/10 | 8/10 | 1/10 |
#[derive(Debug)]
pub enum SimulationError {
    // Very likely to be user errors
    /// Occurs during calculations of any mathematical update steps such as
    /// [Interaction](cellular_raza_concepts::Interaction) between cells.
    CalcError(CalcError),
    /// Error-type specifically related to the [Controller](cellular_raza_concepts::Controller)
    /// trait.
    ControllerError(ControllerError),
    /// An error specific to cell-division events by the
    ///
    /// [Cycle](cellular_raza_concepts::Cycle) trait.
    DivisionError(DivisionError),
    /// Related to the [PhasedDeath](cellular_raza_concepts::CycleEvent::PhasedDeath) event.
    /// This error can only occurr during the
    /// [update_conditional_phased_death](cellular_raza_concepts::Cycle::update_conditional_phased_death)
    /// method.
    DeathError(DeathError),
    /// Enforcing boundary conditions on cells can exhibhit this boundary error.
    BoundaryError(BoundaryError),
    /// Plotting results. See also [cellular_raza_concepts::plotting].
    DrawingError(DrawingError),
    /// Mostly caused by trying to find a voxel by its index.
    /// This error can also occurr when applying too large simulation-steps.
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
}

impl_from_error! {SimulationError,
    (ReceiveError, RecvError),
    (CalcError, CalcError),
    (ControllerError, ControllerError),
    (DivisionError, DivisionError),
    (DeathError, DeathError),
    (BoundaryError, BoundaryError),
    (IndexError, IndexError),
    (IoError, std::io::Error),
    (DrawingError, DrawingError),
    (StorageError, StorageError)
}

impl_error_variant! {SimulationError,
    SendError,
    ReceiveError,
    CalcError,
    ControllerError,
    DivisionError,
    DeathError,
    BoundaryError,
    IndexError,
    IoError,
    DrawingError,
    StorageError
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
