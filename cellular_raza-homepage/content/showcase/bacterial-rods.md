---
title: Bacterial Rods
date: 2024-06-05
math: true
---

## Mathematical Description
To model the spatial mechanics of elongated bacteria, we represent them as a collection of
auxiliary vertices $\\{\\vec{v}\_i\\}$ which are connected by springs in ascending order.
Furthermore, we assume that the cells are flexible described by their stiffness property.
A force $\vec{F}$ interacting between cellular agents determines the radius (thickness) of the
rods and an attractive component can model adhesion between cells.

### Mechanics
In principle we can assign individual lengths $\\{l_i\\}$ and strengths $\\{\gamma\\}\_i$ to each
spring.
The internal force acting on vertex $\\vec{v}\_i$ can be divided into 2 contributions coming from
the 2 springs pulling on it.
In the case when $i=0,N\_\\text{vertices}$, this is reduced to only one internal component.
We denote with $\vec{c}\_{i}$ the connection between two vertices
$$\begin{align}
    \vec{c}\_i = \vec{v}\_{i}-\vec{v}\_{i-1}
\end{align}$$
and can write down the resulting force
$$\\begin{align}
    \vec{F}\_{i,\text{springs}} =
        &-\gamma\_i\left(1 - \\frac{l\_i}{\left|\vec{c}\_i\right|}\right)
        \vec{c}\_i\\\\
        &+ \gamma\_{i+1}\left(1 - \\frac{l\_{i+1}}{\left|\vec{c}\_{i+1}\right|}\right)
        \vec{c}\_{i+1}
\\end{align}$$

In addition to springs between individual vertices $\vec{v}\_i$, we assume that each angle at a
vertex between two other is subject to a stiffening force.
Assuming that $\alpha_i$ is the angle between the connections and
$\vec{d}\_i=\vec{c}\_i/|\vec{c}\_i|$ is the normalized connection,
we can write down the forces acting on vertices $\vec{v}\_i,\vec{v}\_{i-1},\vec{v}\_{i+1}$
$$\begin{align}
    \vec{F}\_{i,\text{stiffness}} &= \eta\_i\left(\pi-\alpha\_i\right)
        \frac{\vec{d}\_i - \vec{d}\_{i+1}}{|\vec{d}\_i-\vec{d}\_{i+1}|}\\\\
    \vec{F}\_{i-1,\text{stiffness}} &= -\frac{1}{2}\vec{F}\_{i,\text{stiffness}}\\\\
    \vec{F}\_{i+1,\text{stiffness}} &= -\frac{1}{2}\vec{F}\_{i,\text{stiffness}}
\end{align}$$
where $\eta\_i$ is the angle stiffness at vertex $\vec{v}\_i$.
We can see that the stiffening force does not move the overall center of the cell in space.
The total force is the sum of external and interal forces.
$$\begin{equation}
    \vec{F}\_{i,\text{total}} = \vec{F}\_{i,\text{springs}}+ \vec{F}\_{i,\text{stiffness}} + \vec{F}\_{i,\text{external}}
\end{equation}$$
and are integrated via
$$\begin{align}
    \partial\_t^2 \vec{x} &= \partial\vec{x} + D\vec{\xi}\\\\
    \partial\_t\vec{x} &= \vec{F}\_\text{total}
\end{align}$$
where $D$ is the diffusion constant and  $\vec{\xi}$ is the wiener process (compare with
[brownian motion](/docs/cellular_raza_building_blocks/struct.Brownian3D.html)).

### Interaction
When calculating forces acting between the cells, we can use a simplified model to circumvent the
numerically expensive integration over the complete length of the rod.
Given a vertex $\vec{v}\_i$ on one cell, we calculate the closest point $\vec{p}$ on the polygonal
line given by the vertices $\\{\vec{w}\_j\\}$ of the interacting cell.
Furthermore we determine the value $q\in[0,1]$ such that
$$\begin{equation}
    \vec{p} = (1-q)\vec{w}\_j + q\vec{w}\_{j+1}
\end{equation}$$
for some specific $j$.
The force is then calculated between the points $\vec{v}\_i$ and $\vec{p}\_i$ and acts on the
vertex $\vec{w}\_i,\vec{w}\_{i+1}$ with relative strength $(1-q)$ and $q$.
$$\begin{align}
    \vec{F}\_{i,\text{External}} = \vec{F}(\vec{v}\_i,\vec{p})
\end{align}$$
For this example, we reused the interaction shape of the [cell-sorting](/showcase/cell-sorting)
example ignoring the species aspect.

### Cycle
To simulate proliferation, we introduce a growth term for the spring lengths $l\_i$
$$\begin{equation}
    \partial\_t l\_i = \mu
\end{equation}$$
which will increase the length of the cell indefenitely unless we impose a condition for the
[division event](/internals/concepts/cell/cycle).
We define a threshold (in our case double of the original length) for the total length of the
cell at which it divides.
To construct a new cell, we cannot simply copy the existing one twice, but we also need to adjust
internal parameters in the process.
The following actions need to be taken for the old and new agent.

1. Assign a new growth rate (pick randomly from uniform distribution in $[0.8\mu\_0,1.2\mu\_0]$
   where $\mu\_0$ is some fixed value)
2. Assign new positions
    1. Calculate new spring lengths
    $$\tilde{l}\_i = l\_i\left(\frac{1}{2} - \frac{r}{\sum\limits\_i l\_i}\right)$$
    2. Calculate middle of old cell
    $$\vec{m} = \frac{1}{N\_\text{vertices}}\sum\limits\_i\vec{v}\_i$$
    3. Calculate positions of new vertices $\vec{w}\_i$
    $$\\begin{align}
        q\_i &= \frac{i}{N\_\text{vertices}}\\\\
        \vec{w}\_{i,\text{new},\pm} &= (1-q\_i)\vec{m} + q\_i(\vec{v}_{\pm\text{start}} - \vec{m})
    \end{align}$$

{{< callout type="warning" >}}
This is a rather rudimentary implementation of how to calculate the new positions of the cells.
To enhance this approach, we could "go along" the existing polygonal line instead of simply
interpolating between middle and either of the end points.
{{< /callout >}}

### Domain
In this simulation example, the domain plays an important role.
The domain consists of an elongated box with reflective boundary conditions.
Cells are initially placed in the left part.
Due to their repulsive potential at short distances, they begin to push each other into the
remaining space.

## Results
### Initial Snapshot
![](/showcase/bacterial-rods/initial.png)

### Movie
<br>
<video controls>
    <source src="/showcase/bacterial-rods/movie.mp4" type="video/mp4">
</video>

### Final Snapshot
![](/showcase/bacterial-rods/final.png)

## Code
The code is part of the examples and can be found in the official github repository under
[bacterial-rods](https://github.com/jonaspleyer/cellular_raza/tree/master/cellular_raza-examples/bacterial_rods).
