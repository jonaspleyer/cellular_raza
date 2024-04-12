---
title: Free Motile Vertex Model
date: 2024-04-11
---

<!-- TODO -->

## Mathematical Description

## Parameters

## Initial State

![](/showcase/free-motile-vertex-model/cells_at_iter_0000100000.png)

## Results & Movie

<video controls>
    <source src="/showcase/free-motile-vertex-model/movie.mp4">
</video>

{{< callout type="info" >}}
Note: Compared to the script which was used to generate this movie, the final result was again speed up by a factor of 3 with the following command:

```bash
ffmpeg -i output.mp4 -filter:v "setpts=PTS/3" movie.mp4
```
{{< /callout >}}

## Code

The code for this simulation and the visualization can be found in the
[examples](https://github.com/jonaspleyer/cellular_raza/tree/master/cellular_raza-examples/kidney_organoid_model)
folder of `cellular_raza`.

