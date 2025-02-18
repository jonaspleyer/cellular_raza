# organoid_turing_growth

To run the simulation from the base folder of the `cellular_raza` workspace use

```bash
cargo run -r --bin cr_organoid_turing_growth
```

or from the folder of this example `cellular_raza-examples/organoid_turing_growth`

```bash
cargo run -r
```

To plot results with the supplied python script install the dependencies found in the
`requirements.txt` file and execute said script from the folder of this example.

```bash
# We are inside cellular_raza-examples/organoid_turing_growth
python src/plotting.py
```

To see more in-depth guides for these examples visit
[cellular-raza.com/showcase/](https://cellular-raza.com/showcase/).
