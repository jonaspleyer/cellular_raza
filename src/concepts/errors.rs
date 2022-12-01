use core::fmt::{Display,Debug};
use core::any::type_name;

use crossbeam_channel::{SendError,RecvError};


// Define a error which can occur during general calculation
#[derive(Debug,Clone)]
pub struct CalcError {
    pub message: String,
}

impl Display for CalcError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}


// Define an error that can occur during boundary calculation
#[derive(Debug,Clone)]
pub struct BoundaryError {
    pub message: String,
}

impl Display for BoundaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}


#[derive(Debug)]
pub enum ErrorVariant {
    SendError(String),
    ReceiveError(RecvError),
    CalcError(CalcError),
    BoundaryError(BoundaryError),
}


// Implement that this error variant actually is an error type
impl std::error::Error for ErrorVariant {}


impl Display for ErrorVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Sender errors
            ErrorVariant::SendError(message) =>
                write!(f, "{}", message),

            // Receiver errors
            ErrorVariant::ReceiveError(recv_error) =>
                write!(f, "{}", recv_error),

            // Errors during calculation
            ErrorVariant::CalcError(calc_error) =>
                write!(f, "{}", calc_error),

            // Errors if cells cannot be kept inside the simulation domain for whatever reason
            ErrorVariant::BoundaryError(boundary_error) =>
                write!(f, "{}", boundary_error),
        }
    }
}


// Sending
impl<T> From<SendError<T>> for ErrorVariant {
    fn from(_err: SendError<T>) -> Self {
        ErrorVariant::SendError(format!("Error receiving object of type {}", type_name::<SendError<T>>()))
    }
}

// Receiving
impl From<RecvError> for ErrorVariant {
    fn from(err: RecvError) -> Self {
        ErrorVariant::ReceiveError(err)
    }
}


// Calculation
impl From<CalcError> for ErrorVariant {
    fn from(err: CalcError) -> Self {
        ErrorVariant::CalcError(err)
    }
}

// Boundary
impl From<BoundaryError> for ErrorVariant {
    fn from(err: BoundaryError) -> Self {
        ErrorVariant::BoundaryError(err)
    }
}
