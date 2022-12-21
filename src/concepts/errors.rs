use core::fmt::{Display,Debug};
use core::any::type_name;

use std::error::Error;// TODO in the future use core::error::Error (unstable right now)

use crossbeam_channel::{SendError,RecvError};


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
                }
            }
        }

        // Also implement the general error property
        impl std::error::Error for $name {}
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

define_errors!(
    (CalcError, "General Calculation Error"),
    (IndexError, "Can occur internally when information is not present at expected place"),
    (BoundaryError, "Can occur during boundary calculation")
);


/// Covers all errors that can occur in this Simulation
/// The errors are listed from very likely to be a user error from almost certainly an internal error.
#[derive(Debug)]
pub enum SimulationError {
    // Very likely to be user errors
    CalcError(CalcError),
    BoundaryError(BoundaryError),

    // Less likely but possible to be user errors
    SendError(String),
    ReceiveError(RecvError),

    // Highly unlikely to be user errors
    IndexError(IndexError),
    IOError(std::io::Error),
}


impl_from_error!(SimulationError, (ReceiveError, RecvError), (CalcError, CalcError), (BoundaryError, BoundaryError), (IndexError, IndexError), (IOError, std::io::Error));
impl_error_variant!(SimulationError, SendError, ReceiveError, CalcError, BoundaryError, IndexError, IOError);


// Implement conversion from Sending error manually
impl<T> From<SendError<T>> for SimulationError {
    fn from(_err: SendError<T>) -> Self {
        SimulationError::SendError(format!("Error receiving object of type {}", type_name::<SendError<T>>()))
    }
}
