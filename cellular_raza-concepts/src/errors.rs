use core::fmt::Display;
use std::error::Error;

macro_rules! define_errors {
    ($(($err_name: ident, $err_descr: expr)),+) => {
        $(
            #[doc = $err_descr]
            #[derive(Debug,Clone)]
            pub struct $err_name(
                #[doc = "Error message associated with "]
                #[doc = stringify!($err_name)]
                #[doc = " error type."]
                pub String,
            );

            impl Display for $err_name {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(f, "{}", self.0)
                }
            }

            impl Error for $err_name {}
        )+
    }
}

/// Error during decomposition of a SimulationDomain into multiple subdomains
#[derive(Clone, Debug)]
pub enum DecomposeError {
    /// Generic error encountered during domain-decomposition
    Generic(String),
    /// [BoundaryError] which is encountered during domain-decomposition
    BoundaryError(BoundaryError),
    /// [IndexError] encountered during domain-decomposition
    IndexError(IndexError),
}

impl Display for DecomposeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Error for DecomposeError {}

impl From<BoundaryError> for DecomposeError {
    fn from(value: BoundaryError) -> Self {
        DecomposeError::BoundaryError(value)
    }
}

define_errors!(
    (SetupError, "Occurs during setup of a new simulation"),
    (CalcError, "General Calculation Error"),
    (
        TimeError,
        "Error related to advancing the simulation time or displaying its progress"
    ),
    // (
    //     DecomposeError,
    //     "Error during decomposition of a SimulationDomain into multiple subdomains"
    // ),
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
    (
        CommunicationError,
        "Error which occurs during sending, receiving or transmitting information between threads"
    ),
    (BoundaryError, "Can occur during boundary calculation"),
    (
        ControllerError,
        "Occurs when incorrectly applying a controller effect"
    ),
    (DrawingError, "Used to catch errors related to plotting"),
    (
        RngError,
        "Can occur when generating distributions or drawing samples from them."
    )
);

impl From<String> for TimeError {
    fn from(value: String) -> Self {
        TimeError(value)
    }
}

impl From<std::io::Error> for DecomposeError {
    fn from(value: std::io::Error) -> Self {
        DecomposeError::BoundaryError(BoundaryError(format!("{}", value)))
    }
}

impl From<CalcError> for SetupError {
    fn from(value: CalcError) -> Self {
        SetupError(format!("{}", value))
    }
}

impl<E> From<plotters::drawing::DrawingAreaErrorKind<E>> for DrawingError
where
    E: Error + Send + Sync,
{
    fn from(drawing_error: plotters::drawing::DrawingAreaErrorKind<E>) -> DrawingError {
        DrawingError(drawing_error.to_string())
    }
}
