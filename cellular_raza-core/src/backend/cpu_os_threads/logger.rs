use serde::{Deserialize, Serialize};

/// A logger that prints every information to the standard output.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PrintLogger;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FileLogger {
    filename: std::path::PathBuf,
}

impl FileLogger {
    fn log_error(&self, error: &dyn std::error::Error) -> std::io::Result<()> {
        todo!()
    }
}

impl indicatif::TermLike for FileLogger {
    fn clear_line(&self) -> std::io::Result<()> {
        todo!()
    }

    fn flush(&self) -> std::io::Result<()> {
        todo!()
    }

    fn move_cursor_down(&self, n: usize) -> std::io::Result<()> {
        todo!()
    }

    fn move_cursor_left(&self, n: usize) -> std::io::Result<()> {
        todo!()
    }

    fn move_cursor_right(&self, n: usize) -> std::io::Result<()> {
        todo!()
    }

    fn move_cursor_up(&self, n: usize) -> std::io::Result<()> {
        todo!()
    }

    fn width(&self) -> u16 {
        todo!()
    }

    fn write_line(&self, s: &str) -> std::io::Result<()> {
        todo!()
    }

    fn write_str(&self, s: &str) -> std::io::Result<()> {
        todo!()
    }
}

#[derive(Clone, Debug)]
pub struct CustomLogger {
    pub clear_line: fn() -> std::io::Result<()>,
    pub flush: fn() -> std::io::Result<()>,
    pub move_cursor_down: fn(usize) -> std::io::Result<()>,
    pub move_cursor_left: fn(usize) -> std::io::Result<()>,
    pub move_cursor_right: fn(usize) -> std::io::Result<()>,
    pub move_cursor_up: fn(usize) -> std::io::Result<()>,
    pub width: fn() -> u16,
    pub write_line: fn(&str) -> std::io::Result<()>,
    pub write_str: fn(&str) -> std::io::Result<()>,
    pub log_error: fn(&dyn std::error::Error) -> std::io::Result<()>,
}

#[derive(Clone, Debug)]
pub enum Logger {
    Print(console::Term),
    File(FileLogger),
    CustomLogger(CustomLogger),
}

impl Default for Logger {
    fn default() -> Self {
        Logger::Print(console::Term::stdout())
    }
}

impl Logger {
    pub fn log_error(&self, error: &dyn std::error::Error) -> std::io::Result<()> {
        match self {
            Logger::Print(printer) => printer.write_line(&format!("ENCOUNTERED ERROR:\n{error}")),
            Logger::File(filelogger) => filelogger.log_error(error),
            Logger::CustomLogger(custom_logger) => (custom_logger.log_error)(error),
        }
    }
}

impl indicatif::TermLike for Logger {
    fn clear_line(&self) -> std::io::Result<()> {
        match self {
            Logger::Print(printer) => printer.clear_line(),
            Logger::File(filelogger) => filelogger.clear_line(),
            Logger::CustomLogger(custom_logger) => (custom_logger.clear_line)(),
        }
    }

    fn flush(&self) -> std::io::Result<()> {
        match self {
            Logger::Print(printer) => printer.flush(),
            Logger::File(filelogger) => filelogger.flush(),
            Logger::CustomLogger(custom_logger) => (custom_logger.flush)(),
        }
    }

    fn move_cursor_down(&self, n: usize) -> std::io::Result<()> {
        match self {
            Logger::Print(printer) => printer.move_cursor_down(n),
            Logger::File(filelogger) => filelogger.move_cursor_down(n),
            Logger::CustomLogger(custom_logger) => (custom_logger.move_cursor_down)(n),
        }
    }

    fn move_cursor_left(&self, n: usize) -> std::io::Result<()> {
        match self {
            Logger::Print(printer) => printer.move_cursor_left(n),
            Logger::File(filelogger) => filelogger.move_cursor_left(n),
            Logger::CustomLogger(custom_logger) => (custom_logger.move_cursor_left)(n),
        }
    }

    fn move_cursor_right(&self, n: usize) -> std::io::Result<()> {
        match self {
            Logger::Print(printer) => printer.move_cursor_right(n),
            Logger::File(filelogger) => filelogger.move_cursor_right(n),
            Logger::CustomLogger(custom_logger) => (custom_logger.move_cursor_right)(n),
        }
    }

    fn move_cursor_up(&self, n: usize) -> std::io::Result<()> {
        match self {
            Logger::Print(printer) => printer.move_cursor_up(n),
            Logger::File(filelogger) => filelogger.move_cursor_up(n),
            Logger::CustomLogger(custom_logger) => (custom_logger.move_cursor_up)(n),
        }
    }

    fn width(&self) -> u16 {
        match self {
            Logger::Print(printer) => printer.width(),
            Logger::File(filelogger) => filelogger.width(),
            Logger::CustomLogger(custom_logger) => (custom_logger.width)(),
        }
    }

    fn write_line(&self, s: &str) -> std::io::Result<()> {
        match self {
            Logger::Print(printer) => printer.write_line(s),
            Logger::File(filelogger) => filelogger.write_line(s),
            Logger::CustomLogger(custom_logger) => (custom_logger.write_line)(s),
        }
    }

    fn write_str(&self, s: &str) -> std::io::Result<()> {
        match self {
            Logger::Print(printer) => printer.write_str(s),
            Logger::File(filelogger) => filelogger.write_str(s),
            Logger::CustomLogger(custom_logger) => (custom_logger.write_str)(s),
        }
    }
}
