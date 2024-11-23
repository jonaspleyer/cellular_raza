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

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
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

<!-- There exist a wide variety of many general-purpose agent-based simulation toolkits which are being
actively applied in a different fields of study [@Abar2017; @Datseris2022; @Wilensky:1999]. -->
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

<!--
# Underlying Assumptions and Internals

## List of Simulation Aspects

`cellular_raza` assumes that all dynamics can be categorized into what we call "simulation
aspects".
They represent cellular processes, interactions, changes of the simulation domain and interactions
with the external environment.

| Aspect | Description | Depends on |
| --- | --- | --- |
| **Cellular Agent** | | |
| `Position` | Spatial representation of the cell | |
| `Velocity` | Spatial velocity of the cell | |
| `Mechanics` | Calculates the next increment from given force, velocity and position. | `Position` and `Velocity` |
| `Interaction` | Calculates force acting between agents. Also reacts to neighbors. | `Position` and `Velocity` |
| `Cycle` | Changes core properties of the cell. Responsible for cell-division and death. | |
| `Intracellular` | Intracellular representation of the cell. | |
| `Reactions` | Intracellular reactions | `Intracellular` |
| `ReactionsExtra` | Couples intra- & extracellular reactions | `DomainReactions` |
| `ReactionsContact` | Models reactions between cells purely by contact | `Position`, `Intracellular` |
| **Simulation Domain** | | |
| `Domain` | Represents the physical simulation domain. | |
| `DomainMechanics` | Apply boundary conditions to agents. | `Position`, `Velocity` |
| `DomainForce` | Apply a spatially-dependent force onto the cell. | `Mechanics` |
| `DomainReactions` | Calculate extracellular reactions and effects such as diffusion. | `ReactionsExtra` |
| **Other** | | |
| `Controller` | Externally apply changes to the cells. | |

## Spatially Localized Interactions

One useful assumption within `cellular_raza` is that each and every interaction is of finite range.
This means that cellular agents only interact with a limited amount of neighbors and close
environment.
Any long-ranged interactions must be the result of a collection of short-ranged interactions.
This assumption enables us to split the simulation domain into chunks and process them individually
although some communication is needed in order to deal with boundary conditions.
In practice, this means that any interaction force should be given a cutoff.
It also means that any interactions which need to be evaluated between agents should in theory scale
linearly with the number of agents $\mathcal{O}(n_\text{agents})$.

## Code Structure

`cellular_raza` consists of multiple crates working in tandem.
It was designed to have clear separations between conceptual choices and implementation details.
This approach allows us to have a greater amount of modularity and flexibility than regular
simulation tools.

These crates act on varying levels of abstraction to yield a fully working numerical simulation.
Since `cellular_raza` functions on different levels of abstraction, we try to indicate this in the
table below.

| crate | Abstraction Level | Purpose |
| --- | --- | --- |
| `cellular_raza` | - | Bundle together functionality of all other crates. |
| `concepts` | High | Collection of (mainly) traits which need to be implemented to yield a full simulation. |
| `core` | Intermediate-High | Contains numerical solvers, storage handlers and more to actually solve a given system. |
| `building_blocks` | Intermediate | Predefined components of cell-agents and domains which can be put together to obtain a full simulation. |
| `examples` | Application | Showcases and introductions to different simulation approaches. |
| `benchmarks` | Application | Performance testing of various configurations. |

## Backends

To numerically solve a fully specified system, `cellular_raza` provides backends.
The functionality offered by a backend is the most important factor in determining the workflow of
the user and how a given simulation is executed.
Currently, we provide the default `chili` backend but hope to extend this collection in the future.
Backends may choose to purposefully restrict themselves to a subset of simulation aspects or a
particular implementation eg. in order to improve performance.

### Chili

The `chili` backend is the default choice for any new simulation.
It generates source code by extensively using
[macros](https://doc.rust-lang.org/reference/macros-by-example.html) and
[generics](https://doc.rust-lang.org/reference/items/generics.html) but will only insert only the
required code according to the specified simulation aspects to numerically integrate these aspects.
Afterwards, the generated code is compiled and run.

Every backend function is implemented generically by hand.
We use [trait bounds](https://doc.rust-lang.org/rust-by-example/generics/bounds.html) to enforce
correct usage of every involved type.
The generated code is restricted to methods of structs and derivations of their components
functionality.
To obatin a fully working simulation, the `chili` backend combines these generic methods with
user-provided and generated types.
The `run_simulation!` macro generates code depending on which type of simulation aspect is activated
by the user.
By employing this combined scheme of generics and macros, we leverage the strong type-system and
Rusts language-specific safety to avoid pitfalls which a purely macro-based approach would yield.

### Other Backends

`cellular_raza` also comes with the `cpu_os_threads` backend which was the first backend created.
It is in the midst of being deprecated and only serves for some legacy usecases.
In the future, we hope to add a dedicated backend named `cara` to leverage GPU-accelerated
(Graphical Processing Unit) algorithms.
-->

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

<!-- # Performance

We present two separate performance benchmarks assessing the computational efficacy of our code.
The interested reader can find more details in the documentation under
[cellular-raza.com/benchmarks/2024-07-sim-size-scaling](https://cellular-raza.com/benchmarks/2024-07-sim-size-scaling).

## Multithreading
One measure of multithreaded performance is to calculate the possible theoretical speedup
given by Amdahl's law [@Rodgers1985] $T(n)$ and its upper limit $S=1/(1-p)$

\begin{align}
    T(n) &= T_0\frac{1}{(1-p) + \frac{p}{n}}
    \label{eq:amdahls-law}
\end{align}

where $n$ is the number of used parallel threads and $p$ is the proportion of execution time which
benefits from parallelization.

Measuring the performance of any simulation will be highly dependent on the specific cellular 
properties and complexity.
We chose the cell sorting example which contains minimal complexity in terms of calculating
interaction between cellular agents.
Any computational overhead which is intrinsic to `cellular_raza` and not related to the chosen
example would thus be more likely to manifest in performance results.
The total runtime of the simulation is of no relevance since we are only concerned with relative
speedup upon using additional resources.
In addition, we fixed the frequency of each processor, to account for power-dependent effects.

This benchmark was run on three distinct hardware configurations.
We fit equation \autoref{eq:amdahls-law} and obtain the parameter $p$ from which the theoretical
maximal speedup $S$ can be calculated.

Thus we obtain the values $S_\text{3700X}=13.64$, $S_\text{3960X}=45.05$ and
$S_\text{12700H}=34.72$.

## Scaling of Simulation Size

Since we consider only locally finite interactions between agents, we are able to make optimizations
which lead to a linear instead of quadratic scaling in the case of fixed-density.
We set out to test this hypothesis and measure the numerical complexity of calculating interactions
between increasing cellular agents.
To do so, we again chose the cell-sorting example for its minimal intrinsic computational overhead
and gradually increased the number of cellular agents and domain size while keeping their density
constant.
Afterwards, we fit the resulting datapoints with a quadratic formula.
It is easily recognizable that the observed scaling agrees with the expected results.

\begin{figure}
    \begin{minipage}{0.5\textwidth}
        \includegraphics{figures/thread_scaling.png}
        \caption{Amdahl's law with increasing amounts of CPU resources.}
        \label{fig:thread-scaling}
    \end{minipage}%
    \begin{minipage}{0.5\textwidth}
        \includegraphics{figures/sim-size-scaling.png}
        \caption{Scaling of the total simulation size.}
    \end{minipage}
\end{figure}

# Discussion

We have shown that `cellular_raza` can be applied in a wide variety of contexts.
It can also serve as a numerical backend for the development of python packages.
We have assessed the multithreaded performance of the implemented algorithms and shown that
sufficiently large simulations can be efficiently parallelized on various machines.
The underlying assumptions predict a linear growth in computational demand with linearly growing
problem size which has been confirmed by our analysis.
-->

<!-- Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.-->

<!-- # Citations -->

<!-- Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }-->

# Acknowledgements

The author(s) declare that financial support was received for the research, authorship, and/or
publication of this article.
JP and CF received funding from FET-Open research and innovation actions grant under the European
Union’s Horizon 2020 (CyGenTiG; grant agreement 801041).

# References

