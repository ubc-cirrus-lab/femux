import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../../")
from plotter import plot

from pathlib import Path

data_dir = str(Path(__file__).resolve().parents[2] / "data" / "azure") + "/"

sample_data = data_dir + "knative_deployment_data/workload_invocation_data_max_{}_{}_apps.pickle"
parsed_data = data_dir + "knative_deployment_data/parsed_singlefunc_data.pickle"
parsed_app_data = data_dir + "knative_deployment_data/parsed_app_data.pickle"
full_workload = data_dir + "knative_deployment_data/{}_workload_parsed_data.pickle"


def plot_inv_per_func(max_containers, num_apps):
    names = ["Full", "70% Peak Util 100 Apps", "80% Peak Util 100 Apps", "80% Peak Util 150 Apps"]
    for i, name in enumerate([full_workload.format("full"), sample_data.format("63", "100"), sample_data.format("72", "100"),
                               sample_data.format("72", "150")]):        
        
        data = pd.read_pickle(name)

        print(data)

        data = data.NumInvocations.tolist()

        data = [round(val, 6) for val in data]
        cdfx = np.sort(data)
        cdfy = np.linspace(1 / len(data), 1.0, len(data))
        plt.set_cmap("cividis")
        linestyle = "-" if i % 2 == 0 else "--"
        plt.plot(cdfx, cdfy, label=names[i], linestyle=linestyle)
    
        plt.xscale("log")

        plt.xlabel("Number of Invocations")
        plt.ylabel("Fraction of Applications")
        plt.legend()
        plt.tight_layout()
        plt.savefig("../output_plots/azure/num_inv_per_app.pdf")

    plt.close()


def plot_containers_per_min(max_containers, num_apps):
    names = ["70% Peak Util 100 Apps", "80% Peak Util 100 Apps", "80% Peak Util 150 Apps"]
    for i, name in enumerate([sample_data.format("63", "100"), sample_data.format("72", "100"),
                               sample_data.format("72", "150")]):        
        df = pd.read_pickle(name)

        inv_lists = df.ContainersPerMin.tolist()
        total_inv_counts = np.zeros(len(inv_lists[0]))
    
        for inv_list in inv_lists:
            total_inv_counts = np.add(total_inv_counts, inv_list)
    
        cdfx = np.sort(total_inv_counts)
        cdfy = np.linspace(1 / len(total_inv_counts), 1.0, len(total_inv_counts))
        plt.plot(cdfx, cdfy, label=names[i])
        plt.xlabel("Num Containers Within a Minute")
        plt.ylabel("Fraction of Minutes")
        plt.legend()
        plt.tight_layout()
        plt.savefig("../output_plots/azure/containers_per_min.pdf".format(max_containers, num_apps))

if __name__ == "__main__":
    num_containers = 63
    num_apps = 100
    plot_inv_per_func(num_containers, num_apps)
    plot_containers_per_min(num_containers, num_apps)