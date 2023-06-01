# Roadmap
The following points outline which features will be worked on in the future.
Even if these features are listed, they may be postponed due to complications or waiting for other features to be figured out before tackling them.
For feature requests, please use [github's issue tracker](https://www.github.com/jonaspleyer/cellular_raza/issues).

## The way to version 0.1
- [ ] Find scalable user API with variadic generics (see [my question on the rust-forum](https://users.rust-lang.org/t/varying-generic-parameters-with-features/93333/58))
- [ ] Stabilize Backend API (try to avoid features, see above)
- [ ] Make it multi-scale: varying time-steps for portions of simulation
    - [ ] Find user Interface for time input
- [ ] Stochastic Processes
- [ ] Better concepts for domain decomposition
    - [ ] Probable solution: Define `SubDomain` which hold a number of `Voxel` (similar to what is realized already in the `MultiVoxelContainer` struct of the `cpu_os_threads` backend.)
    - [ ] Efficiently implement this new concept and benchmark
- [ ] Evaluate usage of associated types for some concepts
    - [ ] Probably useful for `CellularReactions` concept

## Planned for the Future
- [ ] Complete Deserialization of Simulation
    - [ ] Restart Simulation from Snapshot
- [ ] Custom adaptive time steppers
- [ ] Proper error handling with strategies
- [ ] Export Formats other then 1:1 storage (such as vtk files for paraview)

## Possible Directions
- [ ] Purely GPU powered Backend (probably restricted generics)
