---
title: Backends
type: docs
weight: 10
---

To numerically solve a fully specified system, `cellular_raza` provides backends.
The [chili](chili) backend is the default backend while the [cpu-os-threads](cpu-os-threads)
backend was the first backend which is being phased out gradually at the moment.

The functionality offered by a backend is the most important factor in determining the workflow of
the user and how a given simulation is executed.

## Specialization
In the future, we plan on providing backends which are specialized for certain types of cellular
representations.
They will allow us to run simulations on the GPU and apply other performance optimizations.

