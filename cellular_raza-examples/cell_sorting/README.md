# cell_sorting

Please also see the showcase example at https://cellular-raza.com/showcase/cell_sorting.
We provide 3 different variants of cell-sorting implementing different mechanical and interaction
properties.
Their specific functionality is documented at
https://cellular-raza.com/docs/cellular_raza_building_blocks.

| Name | Command | Mechanics Building Block |
|:---:|:---:|:---:|
| Default | `cargo run -r --bin cr_cell_sorting_default` | [`NewtonDamped3D`](https://cellular-raza.com/docs/cellular_raza_building_blocks/struct.NewtonDamped3D.html) |
| Brownian | `cargo run -r --bin cr_cell_sorting_brownian` | [`Brownian3D`](https://cellular-raza.com/docs/cellular_raza_building_blocks/struct.Brownian3D.html) |
| Langevin | `cargo run -r --bin cr_cell_sorting_langevin` | [`Langevin3D`](https://cellular-raza.com/docs/cellular_raza_building_blocks/struct.Langevin3D.html) |

These commands are meant to be run from within the `cellular_raza-examples/cell_sorting/` folder.

To plot results with the supplied python script install the dependencies found in the
`requirements.txt` file and execute said script from the folder of this example.

```bash
# We are inside cellular_raza-examples/cell_sorting
python src/plotting.py
```

To see more in-depth guides for these examples visit
[cellular-raza.com/showcase/](https://cellular-raza.com/showcase/).
