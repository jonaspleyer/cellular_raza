---
title: Building Blocks
weight: 30
---

## Derive Macro
We like to explain this technique along a simple example.
Suppose we have defined a 1D celltype and written its [Mechanics](https://docs.rs/cellular_raza/latest/cellular_raza/concepts/mechanics/trait.Mechanics.html) and [Interaction](https://docs.rs/cellular_raza/latest/cellular_raza/concepts/interaction/trait.Interaction.html) implementation already.
We are not concerned with the details of this implementation.
```rust
use serde::{Serialize,Deserialize};

#[derive(Serialize,Deserialize,Clone)]
struct MyCellType {
    ...
}

impl Mechanics<...> for MyCellType {
    ...
}

impl Interaction<...> for MyCellType {
    ...
}
```
You have run your simulation and analyzed some of its properties but now you would like to include one of the provided methods without having to copy each of its fields and implement all methods yourself.
This can be done by using the derive macro `#[derive(CellularProperty)]`.
Let's say, that we would like to add CellularReactions to our model.
We need to modify `MyCellType` like this
```rust
#[derive(Serialize,Deserialize,Clone,CellularProperties)]
struct MyCellType {
    ...
    #[cellular_properties(CellularReactions)]
    cellular_reactions: MyReactions
}
```
Including these modifications and marking the correct field with `#[cellular_properties(CellularReactions)]` tells the compiler to automatically derive the `CellularReactions` trait for `MyCellType`.
```admonish tip
Do not forget to specify this new field and define parameters when instancing the `MyCellType` object.
```

```admonish info
This feature does not work currently but is on the Todo-list and highly desirable.
```

## Complete Bottom-Up Definition
Now we take full advantage of `cellular_raza`.
You are free to implement any concept on your own types.
This approach is mainly used for tightly integrated models where all parts of the model cannot be seperated from each other.
<!-- TODO write a guide for this -->