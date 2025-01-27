# cell_sorting

Please also see the showcase example at https://cellular-raza.com/showcase/cell_sorting.
We provide 3 different variants of cell-sorting implementing different mechanical and interaction
properties.
Their specific functionality is documented at
https://cellular-raza.com/docs/cellular_raza_building_blocks.

| Name | Command | Mechanics Building Block |
|:---:|:---:|:---:|
| Default | `cargo run -r --bin default` | `NewtonDamped3D` |
| Brownian | `cargo run -r --bin brownian` | `Brownian3D` |
| Langevin | `cargo run -r --bin langevin` | `Langevin3D` |

These commands are meant to be run from within the `cellular_raza-examples/cell_sorting/` folder.
