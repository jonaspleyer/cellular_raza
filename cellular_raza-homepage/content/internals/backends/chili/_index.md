---
title: 🌶️ chili
weight: 10
---

> A modular, reusable, general purpose backend

## Overview
The `chili` backend generally works in 3 phases.

### Code generation
In the first stage, code is generated and inserted into the simulation.
We distinguish between two code-generating approaches.

#### Macros
The `chili` crate makes extensive use of macros in order to build a fully working simulation.
Two macros stand out in particular: [`build_aux_storage`](build_aux_storage) and [`build_communicator`](build_communicator).

#### Generics

