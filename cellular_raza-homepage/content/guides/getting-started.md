---
title: Getting Started
weight: 10
---

## From a Template
We provide two slightly different templates to quickly get started with running your first simulation.
The first template is purely written in Rust while the second provides bindings to Python.
Their usage is slightly different.
We advise users who quickly want to develop new models to opt for the first alternative.
If you are interested in integrating an existing setup into your Python code, choose the second alternative.

### Standalone Binary

## From Scratch

To create a new project from scratch, initialize an empty project with cargo and change to this directory.
```bash
cargo init my-new-project
cd my-new-project
```

Afterwards add `cellular_raza` as a dependency.
```bash
cargo add cellular_raza
```

We can now begin to write the main function of our program.
To do this, open the file `src/main.rs` with your favourite text-editor.
`cellular_raza` is divided into different [layers](AbstractionLayers.md) which are combined within a [backend](Backends.md).
By default, we use the `cpu_os_threads` backend which can be loaded from `prelude.rs`.
```rust
use cellular_raza::prelude::*;
```
Now we can use `cargo` to compile and execute the project in release mode with all possible optimizations
```bash
cargo run --release
```
```admonish tip
Execute the simulation in `--release` mode whenever performance is critical.
Use the debug mode `cargo run` to find bugs in your simulation.
```
The following compilation process might take a while.
Feel free to grab a water or coffee.



<!-- TODO -->
