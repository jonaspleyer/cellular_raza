# Design Principles
`cellular_raza` arose from a personal need which could not be satisfied by other frameworks.
Agent-based models are particularly useful in the context of modeling pattern formation and self-organization.
When properly used and executed, they present a useful restrictions in order to find underlying cellular processes.

## Flexible model properties
A common workflow while building new models might be to:
1. Implement a new model property (eg. new way of handling mechanics, reactions, etc.)
2. Check how this new property affects current results qualitatively
3.  Try to parametrize and quantify
4. Repeat for different model property

When chaning our model description, we actively want to avoid, having to rewrite the corresponding solver everytime a new model property is developed.
This means that we need to define a way to handle a multitude of different implementations without knowing their detail while still obtaining correct results.
`cellular_raza` solves this problem by defining [concepts](DesignPrinciples-Concepts.md) (traits in the rust language) that are responsible for interfacing between the generalized solver which is part of the [backend](Backends).

It is a delicate business on how to design concepts in such a way that they are general enough while still providing useful abstractions.

## No hidden Parameters
A  common problem for modelers in this regime is that the governing mechanisms might not be known and even less are quantitatively parametrized.
This means, we desire a model which is not only low in parameters but can be completely parameter-less in its most minimal version (of course without doing anything and completing instantanously).
This is already a requirement which many modeling frameworks cannot satisfy and basically takes care of everything that is out there.
`cellular_raza` was designed such that all parameters are obvious to the user and can be accessed.
We differentiate between two classes of parameters

### Model Parameters
These parameters describe the underlying physical model itself.
Some examples include proliferation rate, potential strength (of cellular forces) or reaction parameters such as uptake and secretion.

### Hyperparameters
The second type of parameters are concerned with controlling the overall simulation flow.
They typically do not have physical meaning and may result in increased or decreased precision of results when changing the parameter.
A good example is the time-step size.

## Reduce your model
During the development process of a new model it is often desirable to reduce yourself to the most simplistic formulation of your current iteration.
This means it is necessary, to simply deactivate parts of the code .
We handle this by en-/disabling features such that the respective parts of the code related to these model properties do not even show up in the compiled binary.

## Keep your Sanity
### Deterministic Results, even when parallelized
Testing new models and finding possible bugs can be extremely frustrating when the calculated results can not be reproduced deterministically, even when using more than a single thread of execution.
The initial seed is considered a hyperparameter and can be manually adjusted.

### Ordering of objects does not matter
Every update step should be able to be executed simultanously and resolved in a way that is independent of the ordering in which cells or other objects are included and stored.
This means that results of a simulation will not only be deterministic as explained above but also invariant under reordering of elements.
```admonish warning
    The generated unique identifiers of cells and voxels may be an exception!
    This behaviour can be backend-dependent.
```
