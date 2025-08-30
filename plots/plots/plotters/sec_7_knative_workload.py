import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../../")
from plotter import plot

from pathlib import Path

plt.rcParams["pdf.fonttype"] = 42
ax = plt.figure(figsize=(4, 3)).add_subplot(111)
data_dir = str(Path(__file__).resolve().parents[2] / "data") + "/"

sample_data = data_dir + "knative_deployment_data/workload_invocation_data_max_63_100_apps.pickle"
full_workload = data_dir + "knative_deployment_data/full_workload_parsed_data.pickle"


def plot_inv_per_func():
    names = ["Azure Workload", "Knative Subtrace"]
    for i, name in enumerate([full_workload, sample_data]):
        data = pd.read_pickle(name)

        print(data)

        data = data.NumInvocations.tolist()

        data = [round(val, 6) for val in data]
        cdfx = np.sort(data)
        cdfy = np.linspace(1 / len(data), 1.0, len(data))
        linestyle = "-" if i % 2 == 0 else "--"
        ax.plot(cdfx, cdfy, label=names[i], linestyle=linestyle)
    
        ax.set_xscale("log")

        ax.set_xlabel("Number of Invocations")
        ax.set_ylabel("Fraction of Applications")
        ax.legend()
        plt.tight_layout()
        plt.savefig("../output_plots/sec_7_knative_workload.pdf")


if __name__ == "__main__":
    plot_inv_per_func()
