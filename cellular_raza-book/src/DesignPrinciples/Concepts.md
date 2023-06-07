# Concepts

<!-- //! - All traits should work in a continuous fashion
//!     - Use getters/setters to alter internal variables
//!     - Objects calculate increments which can be used in complex solvers
//!     - In the future we want to also calculate errors and have adaptive techniques for solving
//! - Goal: Any panic that can occur should be user-generated
//!     - we want to catch all errors
//!     - Goal: evaluate error type and reduce step-size for solving (or solver altogether) and try again from last breakpoint -->

## Cellular
### CellAgent
### Cycle
```admonish warning
    Deterministic behaviour can only be guaranteed when using the provided `rng` methods.
    If a user opts to use a different `rng` (eg. the thread rng) to obtain random values, results can be flaky.
```
### Interaction
### InteractionExtracellularGradient
### CellularReactions
### Mechanics

## Simulation Domain
### Domain
### Voxel
### ExtracellularMechanics

## Other
### Id
### Error

<!-- TODO>
