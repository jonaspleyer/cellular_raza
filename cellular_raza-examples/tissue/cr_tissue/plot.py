import numpy as np
import matplotlib.pyplot as plt

data = np.array(
    [
        [[[99.3, 0.0]], [[100.7, 200.0]], [[200.0, 200.0]], [[200.0, 0.0]]],
        [[[100.0, 100.0]], [[100.7, 200.0]], [[200.0, 200.0]], [[200.0, 100.7]]],
        [[[100.0, 100.0]], [[100.0, 100.0]], [[200.0, 200.0]], [[200.0, 100.7]]],
    ]
)

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(-3, 203)
    ax.set_ylim(-3, 203)

    n = len(data)
    for i, x in enumerate(data):
        ax.plot(
            [*x[:, 0, 0], x[0, 0, 0]],
            [*x[:, 0, 1], x[0, 0, 1]],
            color="k",
            alpha=(i + 1) / n,
            linestyle="-.",
        )

    plt.show()
