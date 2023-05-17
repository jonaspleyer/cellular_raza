use core::any::type_name;
use core::fmt::{Debug, Display};

use std::error::Error; // TODO in the future use core::error::Error (unstable right now)

use crossbeam_channel::{RecvError, SendError};

macro_rules! define_errors {
    ($(($err_name: ident, $err_descr: expr)),+) => {
        $(
            #[doc = $err_descr]
            #[derive(Debug,Clone)]
            pub struct $err_name {
                pub message: String,
            }

            impl Display for $err_name {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(f, "{}", self.message)
                }
            }

            impl Error for $err_name {}
        )+
    }
}

macro_rules! impl_error_variant {
    ($name: ident, $($err_var: ident),+) => {
        // Implement Display for ErrorVariant
        impl Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(
                        $name::$err_var(message) => write!(f, "{}", message),
                    )+
                    $name::SerdeJsonError(message) => write!(f, "{}", message),
                    $name::QuickXmlError(message) => write!(f, "{}", message),
                    $name::FastXmlDeserializeError(message) => write!(f, "{}", message),
                    $name::SledError(message) => write!(f, "{}", message),
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

        impl From<serde_json::Error> for $name {
            fn from(err: serde_json::Error) -> Self {
                $name::SerdeJsonError(err)
            }
        }

        impl From<quick_xml::Error> for $name {
            fn from(err: quick_xml::Error) -> Self {
                $name::QuickXmlError(err)
            }
        }

        impl From<quick_xml::DeError> for $name {
            fn from(err: quick_xml::DeError) -> Self {
                $name::FastXmlDeserializeError(err)
            }
        }

        impl From<sled::Error> for $name {
            fn from(err: sled::Error) -> Self {
                $name::SledError(err)
            }
        }
    }
}

define_errors!(
    (CalcError, "General Calculation Error"),
    (
        StepsizeError,
        "Error occuring when choosing a non-appropriate stepsize"
    ),
    (DivisionError, "Errors related to a cell dividing process"),
    (
        DeathError,
        "Errors occurring during the final death step of a cell"
    ),
    (
        IndexError,
        "Can occur internally when information is not present at expected place"
    ),
    (
        RequestError,
        "Ask the wrong object for information and receive this error"
    ),
    (BoundaryError, "Can occur during boundary calculation"),
    (
        StorageError,
        "Placeholder for when Database is not compiled."
    ),
    (DrawingError, "Used to catch errors related to plotting")
);

/// Covers all errors that can occur in this Simulation
/// The errors are listed from very likely to be a user error from almost certainly an internal error.
#[derive(Debug)]
pub enum SimulationError {
    // Very likely to be user errors
    CalcError(CalcError),
    RequestError(RequestError),
    StepsizeError(StepsizeError),
    DivisionError(DivisionError),
    DeathError(DeathError),
    BoundaryError(BoundaryError),
    DrawingError(DrawingError),
    StorageError(StorageError),

    // Less likely but possible to be user errors
    SerdeJsonError(serde_json::Error),
    QuickXmlError(quick_xml::Error),
    FastXmlDeserializeError(quick_xml::DeError),
    SledError(sled::Error),
    SerializeError(Box<bincode::ErrorKind>),
    SendError(String),
    ReceiveError(RecvError),

    // Highly unlikely to be user errors
    ParseIntError(std::num::ParseIntError),
    Utf8Error(std::str::Utf8Error),
    IndexError(IndexError),
    IOError(std::io::Error),
    ThreadingError(rayon::ThreadPoolBuildError),
    ConsoleLogError(indicatif::style::TemplateError),
}

impl_from_error! {SimulationError,
    (ReceiveError, RecvError),
    (CalcError, CalcError),
    (RequestError, RequestError),
    (StepsizeError, StepsizeError),
    (DivisionError, DivisionError),
    (DeathError, DeathError),
    (BoundaryError, BoundaryError),
    (IndexError, IndexError),
    (IOError, std::io::Error),
    (SerializeError, Box<bincode::ErrorKind>),
    (ParseIntError, std::num::ParseIntError),
    (Utf8Error, std::str::Utf8Error),
    (DrawingError, DrawingError),
    (StorageError, StorageError),
    (ThreadingError, rayon::ThreadPoolBuildError),
    (ConsoleLogError, indicatif::style::TemplateError)
}

impl_error_variant! {SimulationError,
    SendError,
    ReceiveError,
    RequestError,
    CalcError,
    StepsizeError,
    DivisionError,
    DeathError,
    BoundaryError,
    IndexError,
    IOError,
    SerializeError,
    ParseIntError,
    Utf8Error,
    DrawingError,
    StorageError,
    ThreadingError,
    ConsoleLogError
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

impl<E> From<plotters::drawing::DrawingAreaErrorKind<E>> for DrawingError
where
    E: Error + Send + Sync,
{
    fn from(drawing_error: plotters::drawing::DrawingAreaErrorKind<E>) -> DrawingError {
        DrawingError {
            message: drawing_error.to_string(),
        }
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
