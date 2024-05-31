---
title: ☕ Coding Guidelines
type: docs
weight: 120
---

# Performance & Optimizations
During the development of `cellular_raza` multiple choices have guided the design.
- compile time > runtime
- generality (within reason) > performance
- design concepts with performance in mind
    - what might actual implementations look like later on
    - which implementations might not be possible with the new choices?
- avoid anything that is hidden; make everything as obvious as possible
    - no hidden parameters
    - describe control flow and functionality of macros in particular

# Style Guide
Styling your code should be done by `rustfmt` according to the given `rustfmt.toml` file.
When writing a `macro` or `proc_macro`, developers are encouraged to introduce frequent line-breaks since rustfmt cannot (currently) format code inside these blocks.
Please adhere to the specified standard, unless explicit diversions make sense.

# Documentation
## `proc_macros`
In the current Rust ecosystem, it can be difficult to properly document proc_macros.
This project consists of multiple crates, many of which have proc_macros to support them.
We distinguish them by their name, simply appending `_derive` or `_proc_macro` to them.
Their documentation is contained in the original crate under the `crate::derive` or `crate::proc_macro` path.

Since the rust compiler can not run checks on the proc-macro itself, it most important to provide
extensive compilation and runtime tests and document the functionality of these macros as best as
possible.
