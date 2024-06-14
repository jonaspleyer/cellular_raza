---
title: ⎇ Releases
type: docs
weight: 100
---

## cellular_raza 0.0.10
_11th June 2024_
- change [Interaction](/internals/concepts/cell/interaction) concept to return two force values for
  own and other cell.
- fix some bugs in [chili](/internals/backends/chili) backend (mainly related to
  [Cycle](/internals/concepts/cell/cycle) concept)
- migrate some examples from old `cpu_os_threads` to new [chili](/internals/backends/chili) backend
- change [VertexDerivedInteraction](/docs/cellular_raza_building_blocks/struct.VertexDerivedInteraction.html)
  to use closest point on external polygon

## cellular_raza 0.0.9
_1st June 2024_
- major improvements to the [chili](/internals/backends/chili) backend
    - stabilize main routines and macros
    - tested workflow with actual examples
- stabilize [new domain](/docs/cellular_raza_concepts/domain_new) traits
- added more examples
{{< callout type="info" >}}
From now on releases will become more frequent and for smaller feature additions such that version
0.1.0 can be reached sooner.
{{< /callout >}}

## cellular_raza 0.0.8
_16th February 2024_
- Added documentation where needed
- Fix public interface for cpu_os_threads backend
- Compared to 0.0.7 this new version will almost only fix Documentation and dependencies for the nightly compiler

## cellular_raza 0.0.7
_16th February 2024_
- more experimenting with trait for fluid dynamics
- improved documentation and website
- only minor advancements in backend development

## cellular_raza 0.0.6
_2nd February 2024_
Further development of `chili` backend:
- extend documentation and development of the chili backend
- template simulation contains raw experimentation with this new backend
- Mechanics and Interaction trait working
- concise API for running simulations still missing
- proc-macro crate has been expanded with many useful helper macros
- next version might contain a first working iteration of the chili backend

## cellular_raza 0.0.5
_12th December 2023_
- include some pyo3 fixes (prevented from compiling in 0.0.4)

## cellular_raza 0.0.4
_6th December 2023_
- add readme files
- minor bugfix for pyo3 binding generation in building blocks

## initial commit
_27th August 2022_
