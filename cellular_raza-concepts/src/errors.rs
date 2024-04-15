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

/// For internal use: formats an error message to include a link to the bug tracker on github.
#[macro_export]
#[doc(hidden)]
macro_rules! format_error_message(
    (@function) => {
        {
            fn f() {}
            let name = std::any::type_name_of_val(&f);
            name.strip_suffix("::f").unwrap()
        }
    };
    ($bug_title:expr, $error_msg:expr) => {
        {
            let title = $bug_title.replace(" ", "%20");
            let body = $error_msg.replace(" ", "%20");
            format!("Internal Error in file {} function {}: +++ {} +++ Please file a bug-report: \
                https://github.com/jonaspleyer/cellular_raza/issues/new?\
                title={}&body={}",
                format_error_message!(@function),
                file!(),
                $error_msg,
                title,
                body,
            )
        }
    };
);

mod test_error_messages {
    #[test]
    fn test_link_title() {
        let error_message = crate::format_error_message!(
            "The title of this error message",
            format!("Some long description.")
        );
        let mut parts = error_message.split("https");
        parts.next().unwrap();
        let res = parts.next().unwrap();
        assert_eq!(
            res,
            "://github.com/jonaspleyer/cellular_raza/issues/new?\
                title=The%20title%20of%20this%20error%20message&\
                body=Some%20long%20description."
        );
    }
}
