import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("output.txt", skipinitialspace = True)
    times = np.array([df["3"][df["0"] == i].mean() for i in range(min(df["0"]), 1+max(df["0"]))])

    start = 0
    stop = len(times)

    x_values_fit = np.arange(start+1, stop+1)
    y_values_fit = times[0] / times[start:stop]
    fit_func = lambda s, p: 1/((1-p) + p/s)

    res = sp.optimize.curve_fit(fit_func, x_values_fit, y_values_fit)

    x_values_plot = range(1, len(times)+1)

    plt.plot(x_values_plot, [fit_func(x, *res[0]) for x in x_values_plot], label="Parallel portion: {:3.1f}%".format(100.0 * res[0][0]))

    plt.plot(np.arange(1, len(times)+1), times[0] / times, label="measurement data")
    plt.legend()
    plt.xscale('log')
    plt.savefig("output.png")
    plt.show()
