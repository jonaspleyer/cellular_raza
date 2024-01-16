# Coding Guidelines

## Style Guide
Styling your code should be done by `rustfmt` according to the given `rustfmt.toml` file.
When writing a `macro` or `proc_macro`, developers are encouraged to introduce frequent line-breaks since rustfmt cannot (currently) format code inside these blocks.
Please adhere to the specified standard, unless explicit diversions make sense.

## Documenting `proc_macros`
In the current Rust ecosystem, it can be difficult to properly document proc_macros.
This project consists of multiple crates, many of which have proc_macros to support them.
We distinguish them by their name, simply appending `_derive` or `_proc_macro` to them.
Their documentation is contained in the original crate under the `crate::derive` or `crate::proc_macro` path.
