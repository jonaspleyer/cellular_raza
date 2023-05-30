# Backends
The idea for different backends arose from the idea to support the same concepts on different architextures.
Efficient implementations of eg. GPU solvers require that the simulation state is formulated with certain
GPU-compatible types which limits the flexibility of the overall crate.
To avoid unnecessary restrictive assumptions, multiple backends were envisioned, some more suited for flexible designs
where a user can implement own concepts rather freely and others which are more specialized and thus more restrictive
but may have benefits such as specialized hardware support with speed improvements.
