---
linkTitle: "Internals"
cascade:
    type: docs
prev: /
next: internals/folder/
---

# Structure

`cellular_raza` consists of multiple crates working in tandem.

```mermaid
stateDiagram-v2
    concepts --> core
    concepts --> building_blocks
    concepts --> examples
    concepts --> benchmarks
    core --> examples
    core --> benchmarks
    building_blocks --> examples
    building_blocks --> benchmarks
```

# Development

<video src="cellular_raza-development-gource.webm" controls style="width: minmax(100%, 1280px);">
