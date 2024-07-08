---
title: 3D Cell Sorting
date: 2024-01-10
math: true
---

Cell Sorting is a naturally occuring phenomenon which drives many biological processes.
While the underlying biological reality can be quite complex, it is rather simple to describe such
a system in its most basic form.
The underlying principle is that interactions between cells are specific.

## Mathematical Description

We assume that cells are spherical objects which interact via force potentials.

$$\begin{align}
    \sigma &= \frac{r}{R_i + R_j}\\\\
    V(r) &= V_0 \left(\frac{1}{3\sigma^3} - \frac{1}{\sigma}\right)
\end{align}$$

The values $R_i,R_j$ are the radii of the cells ($i\neq j$) interacting with each other.
For simplification, we can assume that they are identical $R_i=R_j=R$.

Furthermore, we assume that the equation of motion is given by

$$
    \partial^2_t x = F - \lambda \partial_t x
$$

where the first term is the usual force term $F = - \nabla V$ obtained by differentiating the
given potential and the second term is a damping term which arises due to the cells being immersed
inside a viscuous fluid.

{{< callout type="info" >}}
Note that we opted to omit the mass factor on the left-hand side of the previous equation.
This means, that units of $V_0$ and $\lambda$ are changing and they incorporate this property.
{{< /callout >}}

We can assume that interactions between cells are restricted to close ranges and thus enforce a
cutoff $\xi$ for the interaction where the resulting force is identical to zero.
We further assume that cells of different species do not attract each other but do repel.
To describe this behaviour, we set the potential to zero when $r>R_i+R_j$ (ie. $\sigma>1$)
and both cells have distinct species type $s_i$.
In total we are left with

$$
    V_{i,j}(r) =
    \begin{cases}
        0 &\text{ if } r\geq\xi\\\\
        0 &\text{ if } s_i\neq s_j \text{ and } \sigma\geq 1\\\\
        V(r) &\text{ else }
    \end{cases}.
$$

## Parameters

In total, we are left with only 4 parameters to describe our system.

| Parameter | Symbol | Value |
| --- | --- | --- |
| Cell Radius | $R_i$ | $6.0\mu \text{m}$ |
| Potential Strength | $V_0$ | $2\mu\text{m}^2\text{min}^{-2}$ |
| Damping Constant | $\lambda$ | $2\text{min}^{-1}$ |
| Interaction Range | $\xi$ | $1.5 R_i$ |

## Initial State

The following table shows the configuration used to solve the system.
In total, 1600 cells with random initial positions and zero velocity were placed inside the domain.

| Property | Symbol | Value |
| --- | --- | --- |
| Time Stepsize | $\Delta t$ | $0.2\text{min}$ |
| Time Steps | $N_t$ | $10'000$ |
| Domain Size | $L$ | $110\mu\text{m}$ |
| Cells Species 1 | $N_{C,1}$ | $800$ |
| Cells Species 2 | $N_{C,2}$ | $800$ |

This results in a total time of $2000\text{min}=33.33\text{h}$.

![](/showcase/cell_sorting/0000000020.png)

## Result & Movie

After the simulation has finished, the cells have self-organized into connected regions of the same
species.

<video controls>
    <source src="/showcase/cell_sorting/movie.mp4" type="video/mp4">
</video>

![](/showcase/cell_sorting/0000006000.png)

## Code

The code for this simulation and the visualization can be found in the
[examples](https://github.com/jonaspleyer/cellular_raza/tree/master/cellular_raza-examples/cell_sorting)
folder of `cellular_raza`.

