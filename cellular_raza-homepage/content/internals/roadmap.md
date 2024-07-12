---
title: 🎏 Roadmap
type: docs
prev: docs/coding-guidelines
next: docs/
weight: 110
---

The following points outline which features will be worked on in the future.
Even if these features are listed, they may be postponed due to complications or waiting for other features to be figured out before tackling them.
For feature requests, please use [github's issue tracker](https://www.github.com/jonaspleyer/cellular_raza/issues).

## The way to version 0.1
### Simulation Flow
- [ ] Make it multi-scale: varying time-steps for portions of simulation
    - [ ] Find user Interface for time input

### CellAgent (concepts + backend)
- [x] mechanics via force interactions
- [x] proliferation
- [x] death
    - [x] immediate death
    - [x] phased death
    - [x] make it stochastic
- [x] Custom (individual-only) rules
- [x] General contact functions
    - [x] Contact reactions

### Domain (concepts + backend)
- [ ] Environment Fluid Dynamics
    - [x] PDE Diffusion
    - [ ] Lattice Boltzmann
    - [ ] Particles
- [x] Better concepts for domain decomposition
    - [x] Test currently proposed new design
    - [x] Efficiently implement this new concept and benchmark
- [ ] Evaluate usage of associated types for some concepts
    - [ ] `CellularReactions` concept
    - [x] `Domain` concept

### Overall Design
- [x] Parallelization of default backend
- [x] Deterministic results (in default backend, even yield same binary output when changing number of threads)
- [x] Stabilize the new `chili` backend

## Planned for the Future
- [ ] Complete Deserialization of Simulation
    - [ ] Restart Simulation from Snapshot with identical binary output
- [ ] Custom (adaptive) time steppers
- [ ] Proper error handling with strategies
- [ ] Export Formats other then 1:1 storage through (de)serialization (such as vtk files for paraview)
- [ ] Csv file storage support
- [ ] Julia Bindings

## Possible Directions
- [ ] Purely GPU powered Backend (probably restricted generics)
- [x] Python bindings for some predefined models
- [ ] Larger than memory simulations by using `sled` on disk
