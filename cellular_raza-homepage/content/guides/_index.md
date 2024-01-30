---
title: Guides
cascade:
  type: docs
---

All user Guides assume a basic knowledge of the Rust programming language.
To get started with writing Rust please follow the guidance on the [official website](https://www.rust-lang.org).
For an introduction to agent-based modeling, please refer to [wikipedia](https://en.wikipedia.org/wiki/Agent-based_model) and our [review](https://doi.org/10.3389/fphy.2022.968409).
Although beneficial, we do not expect the reader of these guides to be familiar with any computational methods.

## Overview
| Guide | Description |
| --- | --- |
| [Getting Started](getting-started.md) | Step-by-step tutorial on how to write, run and visualize your first simulation |
| [Predefined Cell Models](predefined-cell-models) | Use given cell-models to run a simulation |
| [Building Blocks](building-blocks) | Use predefined building blocks to combine them into a fully working simulation. |
| [Physical Domain](physical-domain) | |

| [Implement your own Concepts](UserGuides-ImplementOwnConcepts.md) | In order to fully take advantage of `cellular_raza`, you can define your own cellular properties and implement them. Learn how to go about this in this guide. |
| [Using Predefined Cellular Modules](UserGuides-PredefinedCellularModules.md) | `cellular_raza` provides some predefined cellular properties. Learn how to use them in your simulation. |
| [Starting from a previous simualtion](UserGuides-StartFromPreviousSimualtion.md) | `cellular_raza` allows to run simulations, stop them and continue running them from the last known savepoint. Learn how you can exploit this functionality. |
