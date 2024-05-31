---
title: 💥 Mechanics
type: docs
weight: 10
math: true
---

The [`Mechanics`](/docs/cellular_raza_concepts/trait.Mechanics.html) and
[`SubDomainMechanics`](/docs/cellular_raza_concepts/domain_new/trait.SubDomainMechanics.html) traits
specify the physical representation of cellular agents inside the simulation domain.
The traits act on 3 distinct types for position, velocity and forces acting on the cell-agents.
In many examples, these types are identical.
The subdomain trait is responsible for keeping cell-agents inside the specified simulation
domain.

# Difference to [Interaction](/internals/concepts/cell/interaction) concept

To describe physical interactions between cells when in proximity of each other, we also provide the
[Interaction](/internals/concepts/cell/interaction) concept.
In contrast, the Mechanics concepts is only concerned with the physical representation and physical
motion of one cell on its own.
Although these two concepts seem to be similar in nature, it can be benefitial to separate them not
only conceptually but also for practical reasons.
For example we can seamlessly change between the
[`Brownian3D`](/docs/cellular_raza_building_blocks/struct.Brownian3D.html) and
[`Langevin3D`](/docs/cellular_raza_building_blocks/struct.Langevin3D.html) struct without having to
alter the currently used Interaction type (if present).

# Examples
A wide variety of cellular repesentations can be realized by this trait.
`cellular_raza` provides some of them in its
[`cellular_raza_building_blocks`](/docs/cellular_raza_building_blocks) crate.

## Point-like Particles
To illustrate how the [`Mechanics`](/docs/cellular_raza_concepts/Mechanics.html) concept works, we
take alook at the most simple representation as point-like particles.
In this case, cells are described by a postion and velocity vector of $n$ (typically $n=2,3$)
dimensions which we will call `VectorN` for simplicity.
$$
    \vec{x} = \begin{bmatrix}
        x_1\\\\
        x_2\\\\
        \vdots\\\\
        x_n
    \end{bmatrix}
    \hspace{1cm}
    \vec{v} = \begin{bmatrix}
        v_1\\\\
        v_2\\\\
        \vdots\\\\
        v_n
    \end{bmatrix}
$$
A third type is also of importance which describes the force acting on the cell.
In our case we can assume the same form as before
$$
    \vec{F} = \begin{bmatrix}
        F_1\\\\
        F_2\\\\
        \vdots\\\\
        F_n
    \end{bmatrix}.
$$
The solver of the chosen [backend](/internals/backends) must be told how to increment position and
velocity of the cell.
This is done by the `calculate_increment` function.
```rust
fn calculate_increment(&self, force: For) -> Result<(Pos, Vel), CalcError> {
    // Do some calculations
    ...
    // Return increments
    Ok((...,...))
}
```
Even for simple point-like particles, we have a variety of options.
If we assume simple newtonian dynamics without any additional stochastic effects, we can write down
the equations of motion
$$\begin{align}
    \dot{\vec{x}} &= \vec{v}\\\\
    \dot{\vec{v}} &= \frac{1}{m}\vec{F}.
\end{align}$$
Note that we assume that our cell has a certain mass, which may also be set to $m=1$ and thus
neglected in some implementations.
```rust
struct MyCell {
    pos: VectorN,
    vel: VectorN,
    mass: f64,
}
```
The [`Mechanics`](/docs/cellular_raza_concepts/Mechanics.html) trait requires some setters and
getters but the main driver sits behind the `calculate_increment` function.
If we implement it with the previous equations, we obtain
```rust
impl Mechanics<VectorN, VectorN, VectorN> for MyCell {
    // Just some getters ..
    fn pos(&self) -> VectorN {
        self.pos.clone()
    }

    fn velocity(&self) -> VectorN {
        self.vel.clone()
    }

    // ... and setters
    fn set_pos(&mut self, pos: &VectorN) {
        self.pos = pos.clone();
    }

    fn set_velocity(&mut self, velocity: &VectorN) {
        self.vel = velocity.clone();
    }

    // Here is the magic
    fn calculate_increment(&self, force: VectorN) -> Result<(VectorN, VectorN), CalcError> {
        let dx = self.vel();
        let dv = 1/self.mass * force;
        Ok((dx, dv))
    }
}
```
If the user decides to leave out the Interaction concept, we will assume a force of zero as
specified by the [`num::Zero`](https://docs.rs/num/latest/num/traits/trait.Zero.html) trait when
numerically solving the equations.
