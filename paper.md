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
 - name: Freiburg Center for Data Analysis, Modeling and AI
   index: 1
date: 13 March 2025
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
Many tools have emerged which are capable of describing cellular systems in various details
[@Pleyer2023; @Abar2017].
While these tools have proven to be effective for targeted research questions,
they often lack the ability to be applied more generically.\
<!-- Nevertheless, core functionalities such as numerical solvers, storage solutions or domain
decomposition methods could be shared between models.-->
General-purpose ABM toolkits on the other hand are designed without specific applications
in mind [@Abar2017; @Datseris2022; @Wilensky:1999].
They are often able to define agents bottom-up and can be a good choice if they allow for the
desired cellular representation.
However, they lack the explicit forethought to be applied in cellular systems
and may not be able to describe every cellular aspect.

In contrast to classical particle simulations, Agent-based models (ABMs) treat every cell individually.
This implies that parameters can vary between agents and that every cell should be traceable
throughout time and space.
In addition, they can describe growth, proliferation, death and many other cellular processes and
should also accurately model cell lineage.
These models live on the mesoscopic scale where the underlying complexity of the problem can
often not be fully attributed to neither intracellular nor extracellular processes.
Their applications include modeling of self-organization and emergent phenomena but they can
also be used to introduce spatial effects into existing population-based models.
In order to address these issues and construct models from first principles without any
assumptions regarding the underlying complexity or abstraction level, we developed
`cellular_raza`.

## Cellular Agent-Based Frameworks

In our previous efforts [@Pleyer2023], we assessed the overall state of modelling toolkits for
individual-based cellular simulations.
These frameworks are designed for specific usages and often  require many parameters which are
unknown or difficult to determine experimentally.
This poses an inherent problem for their applicability and the ability to properly interpret
results.
Few modelling frameworks exist that provide a significant degree of flexibility and customization in
the definition of cell agents.
Chaste [@Cooper2020] allows reuse of individual components , such as ODE and PDE solvers, but is
only partially cell-based.
Biocellion [@Kang2014] supports different cell shapes such as spheres and cylinders, but admits that
their current approach lacks flexibility in the subcellular description.
BioDynaMo [@breitwieser_biodynamo_2022] offers some modularity in the choice of components for
cellular agents, but cannot deeply customize the cellular representation.

# cellular_raza

We distinguish between different simulation aspects, i.e., mechanics, interaction, or cell cycle.
These aspects are directly related to the properties of the cells, domain, or other external
interactions.
The user selects a cellular representation, which can be built from pre-existing building blocks or
 fully customized bottom-up.
'cellular_raza' utilizes macros to generate code contingent on the simulation aspects.
It makes extensive use of generics and provides abstract numerical solvers.
'cellular_raza' encapsulates the inherent complexity of the code generation process, yet enables
users to modify the specifics of the simulation through the use of additional keyword arguments.
Consequently, users are able to fully and deeply customize the representation and behaviour of the
agents.
Each simulation aspect is abstractly formulated as a trait in Rust's type system.
The getting-started guide provides a good entry point and explains every step from building to
running and visualizing.

# Examples

In the following, we present four different examples of how to use `cellular_raza` (see
[cellular-raza.com/showcase](https://cellular-raza.com/showcase)).

## Cell Sorting

Cell sorting is a naturally occurring phenomenon [@Steinberg1963; @Graner1992].
The cellular interaction is specific to their species.
We consider two distinct types represented by soft spheres.
They physically attract each other at close proximity if their species is identical.
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
which grow asymmetrically in the direction of elongation.
Our model describes the physical mechanics of one cell as a collection of multiple vertices
$\vec{v}_i$ which are connected by springs.
Their relative angle $\alpha$ at each connecting vertex introduces a curvature force which is
proportional to $2\tan(\alpha/2)$.
Cells interact via a soft-sphere force potential with short-ranged attraction.
Multiple contributions are calculated between every vertex and the closest point on the
other cells edges.
In addition, the cell cycle introduces growth of the bacteria until it
divides in the middle into two new cells.
This growth is downregulated by an increasing number of neighboring cells.
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
Cells are modeled as soft spheres which take up nutrients from the domain.
By consuming intracellular nutrients, the cell grows continuously and divides upon reaching a
threshold.
Cells are initially placed inside a centered square after which they grow outwards into the
nutrient-rich area.
They are colored bright purple while they are actively growing and dark when not subject to growth
anymore.
A lighter color in the outer domain indicates that more nutrients are available while a dark color
signifies a lack thereof.

\begin{figure}[!h]
    \centering
    \includegraphics[width=0.5\textwidth]{figures/cells_at_iter_0000009800.png}%
    \includegraphics[width=0.5\textwidth]{figures/cells_at_iter_0000060200.png}
    \caption{
        The bacterial colony grows outwards towards the nutrient-rich parts of the domain thus
        forming branches in the process.
    }
\end{figure}

## Semi-Vertex Model for Epithelial and Plant Cells

Vertex models are actively being used to describe mechanical properties of
plant cells [@Merks2011] or organoid structures of epithelial cells [@Fletcher2014; @Barton2017].
We represent cells by a polygonal collection of vertices connected by springs.
An inside pressure pushes vertices outwards, creating perfect hexagonal cells.
Cells are attracting each other but whenever two polygons overlap, a repulsive force acts.
They are placed in a perfect hexagonal grid such that edges and vertices align and assigned growth
rates from a uniform distribution.

\begin{figure}[!h]
    \includegraphics[width=0.5\textwidth]{figures/snapshot-00000000000000000050.png}
    \includegraphics[width=0.5\textwidth]{figures/snapshot-00000000000000020000.png}
    \caption{
        During growth the cells push on each other thus creating small spaces in between them as the
        collection expands.
        These forces also lead to deviations in the otherwise perfect hexagonal shape.
    }
\end{figure}

# Further Information
The full documentation including guides, all examples from above and more is available at
[cellular-raza.com](https://cellular-raza.com/).
`cellular_raza` can also be used as a numerical backend together with the `pyo3` and `maturin`
[@PyO3_Project_and_Contributors_PyO3,@maturin2025] crates.

# Acknowledgements

The author(s) declare that financial support was received for the research, authorship, and/or
publication of this article.
JP and CF received funding from FET-Open research and innovation actions grant under the European
Union’s Horizon 2020 (CyGenTiG; grant agreement 801041).

# References

