---
linkTitle: "Internals"
cascade:
    type: docs
prev: /
next: internals/folder/
---

`cellular_raza` consists of multiple crates working in tandem.

```mermaid
classDiagram
    concepts <|-- core
    concepts <|-- building_blocks
    concepts <|-- examples
    core <|-- examples
    building_blocks <|-- examples
```
