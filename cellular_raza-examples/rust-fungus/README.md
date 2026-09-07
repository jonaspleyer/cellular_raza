<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jonaspleyer/cellular_raza/refs/heads/master/cellular_raza/logos/cellular_raza_dark_mode.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/jonaspleyer/cellular_raza/refs/heads/master/cellular_raza/logos/cellular_raza.svg">
        <img alt="The cellular_raza logo" src="doc/cellular_raza.svg">
    </picture>
</div>

# cellular_raza-template-pyo3
[![License: GPL 2.0](https://img.shields.io/github/license/jonaspleyer/cellular_raza-template-pyo3?style=flat-square)](https://opensource.org/license/gpl-2-0/)
[![CI](https://img.shields.io/github/actions/workflow/status/jonaspleyer/cellular_raza-template-pyo3/CI.yml?label=CI&style=flat-square)](https://github.com/jonaspleyer/cellular_raza-template-pyo3/actions)
![Docs](https://img.shields.io/github/actions/workflow/status/jonaspleyer/cellular_raza-template-pyo3/docs.yml?label=Docs&style=flat-square)

This templates automatically creates documentation under
[jonaspleyer.github.io/cellular_raza-template-pyo3](https://jonaspleyer.github.io/cellular_raza-template-pyo3/).

## Usage

We rely on [pyo3](https://pyo3.rs) and [maturin](https://www.maturin.rs/tutorial) to build this
package.
We require a working virtual environment where the package can be installed.
See [venv](https://docs.python.org/3/library/venv.html) for how to do this on your platform.

```bash
# Compiles and installs the package in the active virtual environment.
maturin develop -r
```

Afterwards, the package can be used in any python script

```python
>>> import cellular_raza_template_pyo3 as crt
```

For more information see the [guide on python bindings](https://cellular-raza.com/guides) for
`cellular_raza`.

## List of files to change before using this template

```bash
[3] ├── Cargo.toml
[1] ├── cellular_raza_template_pyo3
[1] │   └── __init__.py
[ ] ├── docs
[2] │   ├── conf.py
[2] │   ├── index.rst
[ ] │   ├── make.bat
[ ] │   ├── Makefile
[ ] │   ├── references.bib
[ ] │   ├── requirements.txt
[ ] ├── examples
[1] │   └── basic.py
[ ] ├── .github
[ ] │   └── workflows
[ ] │       ├── CI.yml
[ ] │       └── docs.yml
[2] ├── .gitignore
[ ] ├── LICENSE
[ ] ├── make.bat
[2] ├── pyproject.toml
[1] ├── README.md
[ ] ├── requirements.txt
[ ] └── src
[1]     └── lib.rs
```

To use the workflow provided under `.github/workflows/docs.yml`, you need to enable
[publishing github pages from actions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

| File/Folder | |
|---| --- |
| `Cargo.toml` | `name = ...` attributes under `[package]` and `[lib]` |
| `Cargo.toml` | Update the dependency `cellular_raza = ...` to the most recent version. |
| `cellular_raza_template_pyo3` | Folder name and import statement in `__init__.py` |
| `docs/conf.py` | `project = '...'` |
| `docs/conf.py` | Import statement `import cellular_raza_template_pyo3` |
| `docs/index.rst` | First directive `.. cellular_raza_template_pyo3 ...` |
| `docs/index.rst` | Title `cellular_raza_template_pyo3` |
| `examples/basic.py` | `import cellular_raza_template_pyo3 as crt` |
| `.gitignore` | `cellular_raza_template_pyo3/*.so` |
| `.gitignore` | `cellular_raza_template_pyo3/__pycache__` |
| `pyproject.toml` | `name = ...` |
| `pyproject.toml` | `module-name = "..."` |
| `README.md` | Link to documentation |
| `README.md` | Title and remove this table when done. |
| `src/lib.rs` | Name of module `fn cellular_raza_template_pyo3_rs(m: ...` Notice that we use the suffix `_rs` to indicate that this is the rust-specific module. This is optional but needs to be adjusted in the corresponding `cellular_raza_template_pyo3/__init__.py` and `pyproject.toml` files. |
