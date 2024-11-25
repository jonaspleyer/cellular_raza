---
title: 'cellular_raza: Cellular Agent-based Modeling from a Clean Slate'
tags:
  - rust
  - biology
  - agent-based
  - cellular
authors:
  - name: Jonas Pleyer
    orcid: 0009-0001-0613-7978
    affiliation: 1
  - name: Christian Fleck
    affiliation: 1
affiliations:
 - name: Freiburg Center for Data-Analysis and Modeling
   index: 1
date: 01 June 2024
bibliography: paper.bib

---

# Summary

`cellular_raza` is a cellular agent-based modeling framework which allows researchers to construct
models from a clean slate.
In contrast to other agent-based modeling toolkits, `cellular_raza` was designed to be free of
assumptions about the underlying cellular representation.
This enables researchers to build up complex models while retaining full control over every
parameter introduced.
It comes with predefined building blocks for agents and their physical domain to quickly
construct new simulations bottom-up.
Furthermore, `cellular_raza` can be used with the `pyo3` and `maturin` packages and thus act as a
numerical backend to a python package.

# Statement of Need

Agent-based models have become popular in cellular biology
[@Mogilner2016; @Cess2022; @Delile2017; @Delile_Herrmann_Peyrieras_Doursat_2017].
<!-- and many tools have been developed so far to asses specific questions in specialized fields -->
While these tools have proven to be effective for targeted research questions,
they often lack the ability to be applied for multiple distinct use-cases in a more generic context.
Nevertheless, core functionalities such as numerical solvers, storage solutions, domain
decomposition methods and functions to construct these simulations could be shared between models
if written generically.
In order to address this issue and construct models from first principles without any assumptions
regarding the underlying complexity or abstraction level, we developed `cellular_raza`.

# State of Field
## General-Purpose Agent-Based Modeling Toolkits

General-purpose agent-based toolkits are often designed without specific applications in mind
[@Abar2017; @Datseris2022; @Wilensky:1999].
They are often able to define agents bottom-up and can be a good choice if they allow for the
desired cellular representation.
However, they lack the explicit forethough to be applied in cellular systems.
Since they are required to solve a wider range of problems they are not able to make assumptions on
the type of agent or the nature of their interactions and thus miss out on possible
performance optimizations and advanced numerical solvers.

## Cellular Agent-Based Frameworks

In our previous efforts [@Pleyer2023], we assessed the overall state of modelling toolkits for
individual-based cellular simulations.
The frameworks reviewed are all designed for specific use cases and often require a large number of
parameters which are often unknown in practice and difficult to determine experimentally.
This is an inherent problem for the applicability of the software and the ability to properly
interpret results.
Few modelling frameworks exist that provide a significant degree of flexibility and customisation in
the definition of cell agents.
Chaste [@Cooper2020] allows reuse of individual components of its simulation code, such as ODE and
PDE solvers, but is only partially cell-based.
Biocellion [@Kang2014] has support for different cell shapes such as spheres and cylinders, but
admits that their current approach lacks flexibility in the subcellular description.
BioDynaMo [@breitwieser_biodynamo_2022] offers some modularity in the choice of components for
cellular agents, but cannot freely customise the cellular representation.

# cellular_raza

We distinguish between different simulation aspects, e.g.,  mechanics, cell cycle, or cell cycle.
These aspects are directly related to the properties of the cells, domain, or other external
interactions.
The user selects a cellular representation, which can be built from pre-existing building blocks or
a fully customised bottom-up approach, if desired.
'cellular_raza' utilises macros to generate code contingent on the simulation aspects being solved
numerically.
It makes extensive use of generics and provides abstract numerical solvers.
'cellular_raza' hides the inherent complexity of the code generation process, yet enables users to
modify the specifics of the simulation through the use of additional keyword arguments within the
macros.
Consequently, users are able to fully and deeply customise the representation and behaviour of the
agents.
Each simulation aspect is formulated as a trait in Rust's type system, which provides the necessary
abstractions.
The getting-started guide provides a good entry point and explains every step from building, running
to visualising.

# Examples

In the following, we present four different examples how to use `cellular_raza` (see
[cellular-raza.com/showcase](https://cellular-raza.com/showcase)).

## Cell Sorting

Cell sorting is a naturally occurring phenomenon [@Steinberg1963; @Graner1992].
While the underlying biological reality can be quite complex, it is rather simple to describe such
a system in its most basic form.
Fundamentally, any cellular `Interaction` is specific to their species.
We consider two distinct species represented by soft spheres which physically attract each other at
close proximity if their species is identical.
Cells are placed randomly inside a cube with reflective boundary conditions.
In the final snapshot, we can clearly see the phase-separation between the different species.

\begin{figure}[!h]
    \centering
    \includegraphics[width=0.5\textwidth]{figures/cell_sorting_start.png}%
    \includegraphics[width=0.5\textwidth]{figures/cell_sorting_end.png}
    \caption{
        The initial random placement of cells reorders into a phase-separated spatial pattern.
    }
\end{figure}

## Bacterial Rods

Bacteria come in various forms [@Zapun2008; @Young2006] such as elongated shapes [@Billaudeau2017]
which grows asymmetrically in the direction of elongation.
Our model describes the physical mechancis of one cell as a collection of multiple vertices
$\vec{v}_i$ which are connected by springs.
Their relative angle $\alpha$ at each connecting vertex introduces a stiffening force which is
proportional to $2\tan(\alpha/2)$.
Cells interact via a soft-sphere force potential with short-ranged attraction.
Multiple contributions are calculated between every vertex and the closest point on the
other cells edges.
In addition, the cell cycle introduces growth of the bacteria until it
divides in the middle into two new cells.
This growth is downregulated by an increasing number of neighboring cells which is a
phenomenological but effective choice for the transition into the stationary
phase of the bacterial colony.
Cells are placed inside the left-hand side of an elongated box with reflective boundary conditions.
Their colors range from green for fast growth to blue for dormant cells.

\begin{figure}[!h]
    \centering
    \includegraphics[width=0.5\textwidth]{figures/bacterial-rods-0000000025.png}%
    \includegraphics[width=0.5\textwidth]{figures/bacterial-rods-0000007200.png}
    \caption{
        The bacteria extend from the initial placement in the left side towards the right side.
        Their elongated shape and the confined space favour the orientation facing along the growth
        direction.
    }
\end{figure}

## Branching of _Bacillus Subtilis_

Spatio-temporal patterns of bacterial growth such as in _Bacillus Subtilis_ have been studied for
numerous years [@kawasakiModelingSpatioTemporalPatterns1997; @matsushitaInterfaceGrowthPattern1998].
Cells are modeled by soft spheres which interact with the domain by taking up nutrients.
By consuming intracellular nutrients, the cell grows continuously and divides upon reaching a
threshold.
The initial placement of the cells is inside of a centered square.
From there, cells start consuming nutrients and growing outwards towards the nutrient-rich area.
Cells are colored bright purple while they are actively growing and dividing while dark cells are
not subject to growth anymore.
The outer domain is colored by the intensity of present nutrients.
A lighter color indicates that more nutrients are available while a dark color signifies a lack
thereof.

\begin{figure}[!h]
    \centering
    \includegraphics[width=0.5\textwidth]{figures/cells_at_iter_0000028000.png}%
    \includegraphics[width=0.5\textwidth]{figures/cells_at_iter_0000099000.png}
    \caption{
        The bacterial colony grows outwards towards the nutrient-rich parts of the domain thus
        forming branches in the process.
    }
\end{figure}

## Semi-Vertex Model for Epithelial and Plant Cells

Vertex models are a very popular choice in describing multicellular systems.
They are actively being used in great variety such as to describe mechanical properties of plant
cells [@Merks2011] or organoid structures of epithelial cells [@Fletcher2014; @Barton2017].

We represent cells by a polygonal collection of vertices connected by springs.
An inside pressure pushes vertices in an outwards direction.
These two mechanisms by themselves create perfect hexagonal cells.
Cells are attracting each other but in the case where two polygons overlap, a repulsive force acts
between them.
Cells are placed in a perfect hexagonal grid such that edges and vertices align.
Their growth rates are chosen from a uniform distribution.

\begin{figure}[!h]
    \includegraphics[width=0.5\textwidth]{figures/snapshot-00000000000000000050.png}
    \includegraphics[width=0.5\textwidth]{figures/snapshot-00000000000000020000.png}
    \caption{
        During growth the cells push on each other thus creating small spaces in between them as the
        collection expands.
        These forces also lead to deviations in the otherwise perfect hexagonal shape.
    }
\end{figure}

# Acknowledgements

The author(s) declare that financial support was received for the research, authorship, and/or
publication of this article.
JP and CF received funding from FET-Open research and innovation actions grant under the European
Union’s Horizon 2020 (CyGenTiG; grant agreement 801041).

# References

