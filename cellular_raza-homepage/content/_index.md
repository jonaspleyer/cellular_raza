---
toc: false
---

`cellular_raza` is a cellular
[agent-based modeling](https://en.wikipedia.org/wiki/Agent-based_model) framework
which allows researchers to construct models from a clean slate.

In contrast to other agent-based modelling toolkits, `cellular_raza` was designed to be free of
assumptions about the underlying cellular representation.
This enables researchers to build up complex models while retaining full control over every
parameter introduced.

## Selected [Showcases](/showcase)

{{< cards >}}
    {{<
        card link="/showcase/free-motile-vertex-model"
        title="Free Motile Vertex Model"
        image="/showcase/free-motile-vertex-model/cells_at_iter_0000100000.png"
        subtitle="Cells Self-Organizing to Tissue-Like structure"
    >}}
    {{<
        card link="/showcase/cell-sorting"
        title="3D Cell-Sorting"
        image="/showcase/cell_sorting/0000006000.png"
        subtitle="Low-parametric autonomous cell-sorting in 3 dimensions"
    >}}
{{< /cards >}}

## Selected [Benchmarks](/benchmarks)
<!-- TODO this ist just a copy-and-pase and should possibly be automated-->

{{< cards >}}
    {{< card
        link="/benchmarks/2024-02-thread-scaling"
        title="Multithreading"
        subtitle="2024-01-29 We analyze scaling with multiple threads using the `chili` backend and the `cell-sorting` simulation."
        image="/benchmarks/2024-02-thread-scaling/scaling_1.png"
    >}}
{{< /cards >}}

## Explore
<!-- TODO this ist just a copy-and-pase and should possibly be automated-->

{{< cards >}}
  {{< hextra/feature-card
    link="guides"
    title="Guides"
    icon="pencil"
  >}}
  {{< hextra/feature-card
    title="Showcase"
    link="showcase"
    icon="photograph"
  >}}
  {{< hextra/feature-card
    title="Benchmarks"
    link="benchmarks"
    icon="presentation-chart-bar"
  >}}
  {{< hextra/feature-card
    link="internals"
    title="Internals"
    icon="cog"
  >}}

  {{< hextra/feature-card
    link="docs/cellular_raza"
    title="Docs"
    icon="book-open"
  >}}
  {{< hextra/feature-card
    link="publications"
    title="Publications"
    icon="newspaper"
  >}}

{{< /cards >}}
