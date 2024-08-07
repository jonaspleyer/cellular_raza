import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from pathlib import Path
import glob

def load_config_paths(odir: Path = Path("benchmark_results")) -> list[Path]:
    return [Path(p) for p in glob.glob(str(odir) + "/*")]

def load_results(subfolder: str, odir: Path) -> list[dict]:
    config_paths = load_config_paths(odir)
    results_all = []
    for path in config_paths:
        summary = {"path": path, "name": path.name, "subfolder": subfolder}
        results_id = []
        for id_path in glob.glob(str(path) + "/" + subfolder + "/*"):
            id_int = int(str(id_path).split("/")[-1].split(".json")[0])
            file = open(id_path)
            benchmark_result = json.load(file)
            benchmark_result["id"] = id_int
            results_id.append(benchmark_result)
        summary["runs"] = results_id
        results_all.append(summary)
    return results_all

def calculate_throughput(subfolder: str = "thread-scaling", odir: Path = Path("benchmark_results")) -> list:
    """
    Calculates the throughput via
    n_steps / runtime / n_cells_total
    in units 1/second.
    """
    results = load_results(subfolder, odir)
    return [
        [
            (
                r["name"],
                ri["simulation_settings"]["n_threads"],
                ri["simulation_settings"]["n_steps"]\
                    * (ri["simulation_settings"]["n_cells_1"] + ri["simulation_settings"]["n_cells_2"])\
                    / np.array(ri["times"])\
                    / 1e-9
            ) for ri in r["runs"]
        ]
    for r in results]

def calculate_runtime(subfolder: str = "sim-size", odir: Path = Path("benchmark_results")) -> list:
    """
    Calculates the runtime via
    runtime / n_steps
    in units seconds."""
    results = load_results(subfolder, odir)
    return [
        [
            (
                r["name"],
                ri["simulation_settings"]["domain_size"],
                ri["simulation_settings"]["n_cells_1"] + ri["simulation_settings"]["n_cells_2"],
                1e-9 * np.array(ri["times"]) / ri["simulation_settings"]["n_steps"]
            )
            for ri in r["runs"]
        ]
        for r in results
    ]

def get_throughput_dataset(subfolder: str = "thread-scaling", odir: Path = Path("benchmark_results")) -> pd.DataFrame:
    throughput_results = calculate_throughput(subfolder, odir)
    lines = []
    for tr in throughput_results:
        for name, n_threads, throughput in tr:
            lines.append([name, n_threads, np.average(throughput), np.std(throughput)])
    df = pd.DataFrame(lines, columns=["name", "n_threads", "throughput_avg", "throughput_std"])
    df = df.sort_values("n_threads")
    return df

def get_runtime_dataset(subfolder: str = "sim-size", odir: Path = Path("benchmark_results")) -> pd.DataFrame:
    runtime_results = calculate_runtime(subfolder, odir)
    lines = []
    for ri in runtime_results:
        for name, domain_size, n_agents, runtimes in ri:
            lines.append([name, domain_size, n_agents, np.average(runtimes), np.std(runtimes)])
    df = pd.DataFrame(lines, columns=["name", "domain_size", "n_agents", "runtime_avg", "runtime_std"])
    df = df.sort_values("n_agents")
    return df

def plot_runtime(
        entries: list[dict],
        subfolder: str = "sim-size",
        odir: Path = Path("benchmark_results"),
        fit_exponential: bool = True,
        fit_order: int = 2,
    ) -> plt.Figure:
    df = get_runtime_dataset(subfolder, odir)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 25,
        'legend.fontsize': 18,
        'axes.titlesize': 33,
        'lines.markersize': 6,
        'lines.markeredgewidth': 0.5,
        'lines.linewidth': 2,
    })

    def fit_func(x, *params):
        if fit_exponential:
            return np.log(np.sum([p*np.exp(x)**(len(params)-i-1) for i, p in enumerate(params)], axis=0))
        else:
            return np.sum([p*x**(len(params)-i-1) for i, p in enumerate(params)], axis=0)

    fig, ax = plt.subplots(figsize=(12, 9))
    for entry in entries:
        name = entry["name"]
        grp = df[df["name"]==name]
        filt = [
            n in entry.get("sim-sizes", range(len(grp["n_agents"])))
            for n in range(len(grp["n_agents"]))
        ]
        x_values = grp["n_agents"][filt]
        y_values = grp["runtime_avg"][filt]
        # Do individual fits for every curve
        fit_x_values = np.log(x_values) if fit_exponential else x_values
        fit_y_values = np.log(y_values) if fit_exponential else y_values
        p0 = np.zeros(fit_order+1)
        if fit_order > 1:
            p0[-2] = list(y_values)[-1] / list(x_values)[-1]
        popt, pcov = sp.optimize.curve_fit(
            fit_func,
            fit_x_values,
            fit_y_values,
            p0=p0,
            bounds=(0,np.inf),
        )
        color = entry.get("color", "k")
        ax.plot(
            x_values,
            np.exp(fit_func(fit_x_values, *popt)) if fit_exponential else fit_func(fit_x_values, *popt),
            linestyle="--",
            color=color,
        )
        ax.errorbar(
            x_values,
            y_values,
            yerr=grp["runtime_std"][filt],
            label=entry.get("label", entry["name"]),
            color=color,
            fmt="o",
        )
        ax.legend()
        print("=============================================")
        print("| Fitting summary for {} with polynomial of order {}".format(entry.get("label", entry["name"]), len(popt)-1))
        print("|--------------------------------------------")
        print("| Coefficients: ", *["p{}={:.3e}".format(len(popt)-i-1, p) for i, p in enumerate(popt)])
        print("| Variance:     ", *["s{}={:.3e}".format(len(pcov)-i-1, p[i]**0.5) for i, p in enumerate(pcov)])
        print("| Effects at n_agents={}:".format(list(x_values)[-2]))
        for n_x_value in range(len(x_values)):
            x_value = list(x_values)[n_x_value]
            fit_x_value = list(fit_x_values)[n_x_value]
            params = [[popt[i] if i==j else 0 for i in range(len(popt))] for j in range(len(popt))]
            contrib = np.array([fit_func(fit_x_value, *param) for param in params])
            if fit_exponential:
                contrib = np.exp(contrib)
            relative_contrib = contrib / np.sum(contrib)
            print(
                "| Contribution at n_agents={:10}: ".format(x_value),
                *["c{}={:.4e}".format(len(popt)-i-1,c) for i, c in enumerate(contrib)]
            )
            print(
                "| Relative                            ", "",
                *["r{}={:9.1f}%".format(len(popt)-i-1, 100*r) for i, r in enumerate(relative_contrib)]
            )

    # Set some options
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title("Scaling with Problem Size")
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Runtime [s/step]')
    fig.tight_layout()
    fig.savefig(str(odir) + "/sim-size-scaling.png")
    return fig

def plot_throughput(
        entries: list[dict],
        subfolder: str = "thread-scaling",
        odir: Path = Path("benchmark_results"),
    ) -> plt.Figure:
    df = get_throughput_dataset(subfolder, odir)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 25,
        'legend.fontsize': 18,
        'axes.titlesize': 33,
        'lines.markersize': 6,
        'lines.markeredgewidth': 0.5,
        'lines.linewidth': 2,
    })

    fig, ax = plt.subplots(figsize=(12, 9))
    for i, entry in enumerate(entries):
        name = entry["name"]
        grp = df[df["name"]==name]

        filt = np.array([n in entry["threads"] for n in grp["n_threads"]]) if "threads" in entry.keys() else np.repeat(True, len(grp["n_threads"]))
        x_values = grp["n_threads"][filt]
        y_values = grp["throughput_avg"][filt]

        # Do fit
        def fit_func(n, T, p):
            return T / (1 - p + p / n)
        popt, pcov = sp.optimize.curve_fit(
            fit_func,
            x_values,
            y_values,
            p0=(list(grp["throughput_avg"])[0], 1),
            bounds=[(0, 0), (np.inf, 1)],
        )
        color = entry.get("color", "k")

        # FIlter plotted values depending on threads key of entry
        ax.plot(
            grp["n_threads"][filt],
            fit_func(np.array(grp["n_threads"]), *popt)[filt],
            linestyle="--",
            color=color,
        )
        label = entry.get("label", entry["name"])
        ax.errorbar(
            x_values,
            y_values,
            yerr=grp["throughput_std"][filt],
            label="{} $p={:.2f}\pm {:.2f}$%".format(label, 100*popt[1], 100*(pcov[1][1]**0.5)),
            color=color,
            fmt="o",
        )
        ax.legend()
        ax.set_title("Scaling with Threads")
        ax.set_xlabel("Number of Threads")
        ax.set_ylabel("Throughput [steps/s/cell]")
        print("{} S = {:.2f} +- {:.2f}2".format(label, 1/(1-popt[1]), pcov[1][1]**0.5/(1-popt[1])**2))
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(str(odir) + "/thread_scaling.png")
    return fig

plot_runtime(entries=[
    {
        "name": "12700H-at-2000MHz",
        "label": "12700H @2GHz",
        "color": "#ff6361",
    },
    {
        "name": "3960X-at-2000MHz",
        "label": "3960X @2GHz",
        "color": "#58508d",
    },
    {
        "name": "3700X-at-2000MHz",
        "label": "3700X @2GHz",
        "color": "#003f5c",
    },
],
    fit_order=2,
    fit_exponential=False,
)
plot_throughput(entries = [
    {
        "name": "3700X-at-2200MHz",
        "label": "3700X @2.2GHz",
        "color": "#003f5c",
        "threads": list(range(16)),
    },
    {
        "name": "3960X-at-2000MHz",
        "label": "3960X @2GHz",
        "color": "#58508d",
        "threads": list(range(46)),
    },
    {
        "name": "12700H-at-2000MHz",
        "label": "12700H @2GHz",
        "color": "#ff6361",
    }
])
