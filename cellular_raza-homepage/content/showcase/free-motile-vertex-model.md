---
title: Free Motile Vertex Model
date: 2024-04-11
math: true
---

<!-- TODO -->
Vertex models are a very popular choice in describing multicellular systems.
They are actively being used in great variety such as to describe mechanical properties of plant
cells[<sup>1</sup>](#references) or organoid structures of epithelial
cells[<sup>2,3</sup>](#references).

## Mathematical Description
In this model, we are only concerned with cellular forces and their representation in space.
One single cell-agent can be described by a collection of (ordered) vertices which in turn also
allows for a dual description in terms of edges.

$$\\begin{align}
    \\{\vec{v}\_i\\}\_{i=0\\dots n}\\\\
    \vec{v}\_i = \begin{bmatrix}v_{i,0}\\\\v_{i,1}\end{bmatrix}
\\end{align}$$

In the following text, we assume that vertices are always ordered (clockwise or anti-clockwise)
and this ordering is identical for every cell in our simulation.

### Mechanics
Every vertex is connected to its next neighbours in order via springs with an associated length
$d$ and spring constant $\\gamma$.
The potential used to calculate the force $F_i$ acting along the edges of the cell between vertex
$i$ and $i+1$ is given by

$$\\begin{align}
    \vec{F}\_{\text{edges},i} &= - \\gamma \\left(|\vec{v}\_i - \vec{v}\_{i+1}| - d\\right)
    \frac{\vec{v}\_i - \vec{v}\_{i+1}}{|\vec{v}\_i - \vec{v}\_{i+1}|}\\\\
    %V\_\text{edges} &= \sum\limits\_{i=0}^n \\frac{\\gamma}{2}\left(d\_i - d\right)^2
\\end{align}$$

where $d_i = |\vec{v}\_i - \vec{v}\_{i+1}|$ is the distance between individual vertices.

From the length of the individual edges, we can determine the total 2D volume $V$ of the cell when
the equilibrium configuration of a perfect hexagon is reached.

$$\\begin{equation}
    V = d^2\sum\limits_{i=0}^{n-1}\frac{1}{2\sin(\pi/n)}
\\end{equation}$$

However, since the individual vertices are mobile, we require an additional mechanism which
simulates a central pressure $P$ depending on the currently measured volume $\tilde{V}$.
This area can be calculated by summing over the individual areas of the triangles given by
two adjacent vertices and the center point $\vec{c}=\sum\_i\vec{v}\_i/(n+1)$.
They can be calculated by using the parallelogramm formula

$$\\begin{align}
    \tilde{V}\_i &=
    \det\\begin{vmatrix}
        \vec{v}\_{i+1} - \vec{c} & \vec{v}\_i - \vec{c}
    \\end{vmatrix}\\\\
    &= \det\\begin{pmatrix}
        (\vec{v}\_{i+1} - \vec{c})\_0 & (\vec{v}\_{i} - \vec{c})\_0\\\\
        (\vec{v}\_{i+1} - \vec{c})\_1 & (\vec{v}\_{i} - \vec{c})\_1
    \\end{pmatrix}\\\\
    \tilde{V} &= \sum\limits_{i=0}^{n-1}\tilde{V}\_i
\\end{align}$$

The resulting force then points from the center of the cell $\vec{c}$ towards the
individual vertices $\vec{v}\_i$.

$$\\begin{align}
    \vec{F}\_{\text{pressure},i} = P\\left(V-\\tilde{V}\\right)\frac{\vec{v}\_i - \vec{c}}{|\vec{v}\_i - \vec{c}|}
\\end{align}$$

These mechanical considerations alone are enough to yield perfect hexagonal configurations for
individual cells without any interactions.
If we also take into account an external force acting on the cell, the total force acting on the
individual vertices $\vec{v}\_i$ can be calculated via

$$\\begin{equation}
    \vec{F}\_{\text{total},i} = \vec{F}\_{\text{external},i} + \vec{F}\_{\text{edges},i}
        + \vec{F}\_{\text{pressure},i}
\\end{equation}$$

### Interaction

## Parameters

## Initial State

![](/showcase/free-motile-vertex-model/cells_at_iter_0000100000.png)

## Results & Movie

<video controls>
    <source src="/showcase/free-motile-vertex-model/movie.mp4">
</video>

{{< callout type="info" >}}
Note: Compared to the script which was used to generate this movie, the final result was again speed up by a factor of 3 with the following command:

```bash
ffmpeg -i output.mp4 -filter:v "setpts=PTS/3" movie.mp4
```
{{< /callout >}}

## Code

The code for this simulation and the visualization can be found in the
[examples](https://github.com/jonaspleyer/cellular_raza/tree/master/cellular_raza-examples/kidney_organoid_model)
folder of `cellular_raza`.

## References
\[1\] R. M. H. Merks, M. Guravage, D. Inzé, and G. T. S. Beemster, “VirtualLeaf: An Open-Source Framework for Cell-Based Modeling of Plant Tissue Growth and Development      ,” Plant Physiology, vol. 155, no. 2. Oxford University Press (OUP), pp. 656–666, Feb. 01, 2011. [doi: 10.1104/pp.110.167619](https://doi.org/10.1104/pp.110.167619).<br>
\[2\] A. G. Fletcher, M. Osterfield, R. E. Baker, and S. Y. Shvartsman, “Vertex Models of Epithelial Morphogenesis,” Biophysical Journal, vol. 106, no. 11. Elsevier BV, pp. 2291–2304, Jun. 2014. [doi: 10.1016/j.bpj.2013.11.4498](https://doi.org/10.1016/j.bpj.2013.11.4498).<br>
\[3\] D. L. Barton, S. Henkes, C. J. Weijer, and R. Sknepnek, “Active Vertex Model for cell-resolution description of epithelial tissue mechanics,” PLOS Computational Biology, vol. 13, no. 6. Public Library of Science (PLoS), p. e1005569, Jun. 30, 2017. [doi: 10.1371/journal.pcbi.1005569](https://doi.org/10.1371/journal.pcbi.1005569).
