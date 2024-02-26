use std::fs::File;
#[cfg(feature = "tracing")]
use tracing::instrument;
#[cfg(feature = "tracing")]
use tracing_subscriber::{filter, prelude::*};

#[cfg_attr(feature = "tracing", instrument)]
fn asdo() {
    print!("Testaaa");
}

// A layer that logs events to stdout using the human-readable "pretty"
// format.
fn main() {
    #[cfg(feature = "tracing")]
    {
        let stdout_log = tracing_subscriber::fmt::layer()
            .pretty()
            .with_line_number(true)
            .with_level(true);

        // A layer that logs events to a file.
        let file = File::create("debug.log");
        let file = match file {
            Ok(file) => file,
            Err(error) => panic!("Error: {:?}", error),
        };
        let debug_log = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(file);

        tracing_subscriber::registry()
            .with(
                stdout_log
                    // Add an `INFO` filter to the stdout logging layer
                    .with_filter(filter::LevelFilter::INFO)
                    // Combine the filtered `stdout_log` layer with the
                    // `debug_log` layer, producing a new `Layered` layer.
                    .and_then(debug_log),
            )
            .init();

        println!("Test");

        // This event will *only* be recorded by the metrics layer.
        // tracing::info!(target: "metric s::cool_stuff_count", value = 42);
        let span = tracing::span!(tracing::Level::TRACE, "my span 1");
        let _enter = span.enter();
        let span = tracing::span!(tracing::Level::TRACE, "my span 2");
        let _enter = span.enter();
        let span = tracing::span!(tracing::Level::TRACE, "my span 3");
        let _enter = span.enter();
        let span = tracing::span!(tracing::Level::TRACE, "my span 4");
        let _enter = span.enter();

        tracing::info!("target");

        // This event will only be seen by the debug log file layer:
        // tracing::debug!("this is a message, and part of a system of messages");

        // This event will be seen by both the stdout log layer *and*
        // the debug log file layer, but not by the metrics layer.
        // tracing::error!("the message is a warning about danger!");
    }
}
