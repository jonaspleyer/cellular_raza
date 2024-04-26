import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

def ode(t, y, p):
    n_compartments, delay, production, degradation, sink = p
    delays = delay * (y[0:n_compartments] - y[1:n_compartments+1])
    dy = 0 * y
    dy[1:n_compartments+1] = delays
    dy[0] = production - degradation * y[0]
    dy[-1] -= sink * y[-1]
    return dy

if __name__ == "__main__":
    n_compartments = 10

    y0 = np.zeros(n_compartments + 1)
    t_span = (0, 200)

    delay = 0.1
    production = 1.0
    degradation = 0.2
    sink = 0.05
    p = (n_compartments, delay, production, degradation, sink)

    res = sp.integrate.solve_ivp(ode, t_span, y0, args=(p,))
    names = ["compartment {}".format(i) for i in range(res.y.shape[0]-1)] + ["final"]

    plt.plot(res.t, res.y.T, label=names)
    plt.legend()
    plt.show()

