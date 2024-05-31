---
title: 📦 Storage
weight: 30
---

`cellular_raza` assumes that every cell-agent and domain component can be serialed via the
[serde](https://serde.rs/) crate.
This allows us to create full snapshots of an entire simulation which can in principle be used to
load previously executed simulations and continue them.
Said feature is currently not available but part of our [roadmap](/internals/roadmap).
Furthermore, users retain full transparency of every cellular and domain parameter.

# Options
`cellular_raza` provides multiple storage options:

- json
- xml
- sled
- sled (temp)

Of the listed options, all save simulation results to the disk while the "sled (temp)" option
erases any created files after it has commenced.

