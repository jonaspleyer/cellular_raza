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
| [Physical Domain](physical-domain) | How does the [domain](/internals/concepts/domain) in which the cells live work? You will learn what responsibilities are taken over by the domain. |
| [Load a previous Simulation](load-previous-simulation) | Learn how to load data from a previous simulation snapshot and continue to run the simulation. This can be useful when doing complex parameter screening in high-dimensional spaces. |
