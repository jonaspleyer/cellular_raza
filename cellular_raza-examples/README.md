# Examples

Some of these examples will be explained in more detail on the documentation website
[cellular_raza.com](https://cellular-raza.com).
They do not only serve as illustrative examples of what `cellular_raza` is capable of but also as a
reliable method for continuous integration, testing and compatibility.

## Overview

| Name                          | Execution Command                                 | Backend |
|:---                           |:---                                               |:---:|
| `bacteria_population`         | `cargo run -r --bin cr_bacteria_population`       | 🐧 |
| `bacterial_branching`         | `cargo run -r --bin cr_bacterial_branching`       | 🌶️ |
| `bacterial_rods`              | `cargo run -r --bin cr_bacterial_rods`            | 🌶️ |
| `cell_sorting` Default        | `cargo run -r --bin cr_cell_sorting_default`      | 🌶️ |
| `cell_sorting` Brownian       | `cargo run -r --bin cr_cell_sorting_brownian`     | 🌶️ |
| `cell_sorting` Langevin       | `cargo run -r --bin cr_cell_sorting_langevin`     | 🌶️ |
| `cellular_raza-template-pyo3` | -                                                 | 🌶️ |
| `cellular_raza-template`      | `cargo run -r --bin cellular_raza-template`       | 🌶️ |
| `diffusion`                   | Example only used for development.                |    |
| `getting-started`             | `cargo run -r --bin cr_getting_started`           | 🌶️ |
| `homepage-training`           | -                                                 | 🌶️ |
| `immersed_boundary`           | `cargo run -r --bin cr_immersed_boundary`         | 🌶️ |
| `organoid_turing_growth`      | `cargo run -r --bin cr_organoid_turing_growth`    | 🐧 |
| `pool_model_pyo3`             | -                                                 | 🐧 |
| `puzzle`                      | `cargo run -r --bin cr_puzzle_cells`              | 🌶️ |
| `semi_vertex`                 | `cargo run -r --bin cr_semi_vertex`               | 🌶️ |
| `sender-receiver`             | `cargo run -r --bin cr_sender_receiver`           | 🐧 |
| `ureter_signalling`           | `cargo run -r --bin cr_ureter_signalling`         | 🐧 |

Every binary example can also be run by navigating to the respective subfolder and executing `cargo
run -r`.

# External
## cr_mech_coli
This project models rod-shaped bacteria similar to the
[bacterial_rods](https://cellular-raza.com/showcase/bacterial-rods) example.
It aims to estimate parameters, generate near-realistic microscopic images and generate data to
improve existing cell-segmentation and tracking tools.
Find it at [github.com/jonaspleyer/cr_mech_coli](https://github.com/jonaspleyer/cr_mech_coli).

## cr_trichome
This project models trichome formation on top of arabidopsis thaliana leaves.
Find it at [github.com/jonaspleyer/cr_trichome](https://github.com/jonaspleyer/cr_trichome).

## Auophagy Project
This project considers ATG proteins surrounding a cargo and models their binding process to it.<br>
[github.com/jonaspleyer/2023-autophagy](https://github.com/jonaspleyer/2023-autophagy)
