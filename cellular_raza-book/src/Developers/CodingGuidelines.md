# Coding Guidelines

## Documenting `proc_macros`
In the current Rust ecosystem, it can be difficult to properly document proc_macros.
This project consists of multiple crates, many of which have proc_macros to support them.
We distinguish them by their name, simply appending `_derive` or `_proc_macro` to them.
Their documentation is contained in the original crate under the `crate::derive` or `crate::proc_macro` path.
