//! https://docs.rs/tracing/latest/tracing/

/* use std::io::Write;

pub struct FileLogger {
    pub id: u64,
    pub output_file: std::fs::File,
}

impl tracing::Subscriber for FileLogger {
    #[allow(unused)]
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        true
    }

    #[allow(unused)]
    fn new_span(&self, span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(self.id)
    }

    fn record(&self, span: &tracing::span::Id, values: &tracing::span::Record<'_>) {
        self.output_file.write(buf).unwrap()
    }

    fn record_follows_from(&self, span: &tracing::span::Id, follows: &tracing::span::Id) {
        todo!()
    }

    fn event(&self, event: &tracing::Event<'_>) {

    }

    fn enter(&self, span: &tracing::span::Id) {
        todo!()
    }

    fn exit(&self, span: &tracing::span::Id) {
        todo!()
    }
}*/
