---
title: Getting Started
weight: 10
---

## Using Templates
We provide two slightly different templates to quickly get started with running your first
simulation.
The [cell_sorting](https://github.com/jonaspleyer/cellular_raza-template) template is purely
written in Rust while
[cell_sorting-pyo3](https://github.com/jonaspleyer/cellular_raza-template-pyo3) provides Python
bindings in addition.
These templates can serve as a starting point but are not representative of the variability which
`cellular_raza` offers (see [showcase](/showcase)).

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

For now, we only implement physical interactions via the
[Mechanics](/internals/concepts/cell/mechanics) and
[interaction](/internals/concepts/cell/interaction) simulation aspects.
We can quickly build simulations by combining already existing [building_blocks](building-blocks)
with the [CellAgent](/docs/cellular_raza-concepts/derive.CellAgent.html) derive macro.

We begin by importing all necessary 
```rust
#[derive(Clone, CellAgent)]
struct MyCell {
    #[Mechanics]
    mechanics: NewtonDamped3D,
    #[Interaction]
    interaction: MorsePotential,
}
```


We can now begin to write the main function of our program.
To do this, open the file `src/main.rs` with your favourite text-editor.
`cellular_raza` is divided into different [layers](AbstractionLayers.md) which are combined within a [backend](Backends.md).
By default, we use the `cpu_os_threads` backend which can be loaded from `prelude.rs`.
```rust
use cellular_raza::prelude::*;
```

<!--  Now we can use `cargo` to compile and execute the project in release mode with all possible optimizations
```bash
cargo run --releaseHH
```

Execute the simulation in `--release` mode whenever performance is critical.
Use the debug mode `cargo run` to find bugs in your simulation. -->

The following compilation process might take a while.
Feel free to grab a water or coffee.



<!-- TODO -->
