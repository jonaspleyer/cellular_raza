#/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def lennard_jones_force(r, p):
    return 4.0 * p[0] / r * (12.0 * (p[1]/r)**(12.0) - p[2] * (p[1]/r)**(p[2]))

if __name__ == "__main__":
    fig, ax = plt.subplots()

    x = np.linspace(0.1, 10.0, 200)
    r = 2.0
    p1 = [1.0, r/2**(1/6), 0.5]
    p2 = [1.0, r/2**(1/6), 2.0]
    p3 = [1.0, r/2**(1/6), 8.0]

    y1 = lennard_jones_force(x, p1)
    y2 = lennard_jones_force(x, p2)
    y3 = lennard_jones_force(x, p3)

    ax.plot(x, y1, label="1")
    ax.plot(x, y2, label="2")
    ax.plot(x, y3, label="3")

    ax.legend()

    ax.set_ylim([min(y3), -min(y3)])
    plt.show()