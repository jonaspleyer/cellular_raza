---
title: ⏳ Time
weight: 20
---

Every simulation requires a handler for the time configuration.
The [`TimeStepper`](/docs/cellular_raza_core/time/trait.TimeStepper.html) trait determines how
simulation time is advanced.
It can also yield visual feedback on how far the current simulation has propagated by initializing
and updating a progress bar.

# Multi-Scale

`cellular_raza` does currently not support multi-scale methods for numerical integration.
We plan on supporting this feature in the future.
It will most likely require a change to currently existing methods.
See also our [roadmap](/internals/roadmap).

# Adaptive

We currently do not support adaptive solving where the next time-step is determined by the previous
increment but plan on exploring this option in the future.
See also our [roadmap](/internals/roadmap).
