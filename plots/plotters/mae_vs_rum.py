import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plots.plotters.RUM_plotter import gen_result_df, objective_function

from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data" / "azure") + "/"
save_path= str(Path(__file__).parents[1] / "output_plots") + "/"

mae_save_path = data_dir + "results/mae/{}_maes.pickle"

os.makedirs(save_path + "mae_vs_rum/", exist_ok=True)

def plot_metric(forecasters, result_df):   
    args = dict()
    args["x_label"] = "Error"
    args["y_label"] = "Fraction of Applications"
    args["file_name"] = save_path + "mae_vs_rum/comp.pdf"
    args["log"] = True
    args["forecaster_num"] = 0

    for forecaster in forecasters:
        result_df["Metric"] = result_df.apply(lambda x : objective_function(forecaster, x.NumColdStarts, x.MemoryUsed, 
                                                                                x.MemAllocated, x.SkipBlocks, app_level=True), axis=1)
        
        mae_df = pd.read_pickle(mae_save_path.format(forecaster))
        result_df = result_df.merge(mae_df, on="HashApp")
        print(result_df)
    
        vals = list(result_df.Metric.to_list())
        args["label"] = "{}-RUM".format(forecaster)
        plot(vals, args=args)
        
        mae_vals = list(np.array(result_df.MAE.to_list()).flatten())
        args["label"] = "{}-MAE".format(forecaster)
        plot(mae_vals, args=args)

        result_df.drop("MAE", axis=1, inplace=True)



def plot(data, args):
    data = [round(val, 6) for val in data]
    cdfx = np.sort(data)
    cdfy = np.linspace(1 / len(data), 1.0, len(data))
    
    plt.set_cmap("cividis")
    linestyle = "-" if args["forecaster_num"] % 2 == 0 else "--"
    
    plt.plot(cdfx, cdfy, label=args['label'], linestyle=linestyle) 
    if "log" in args and args["log"] == True:
        plt.xscale("log")

    plt.xlabel(args["x_label"].title())
    plt.ylabel(args['y_label'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(args['file_name'])


if __name__ == "__main__":
    forecasters = ["FFT_10", "AR"]
    
    block_size = 504
    percentage = 100
    data_split = "train"
    data_desc = "{}_{}_percent_{}".format(block_size, percentage, data_split)

    result_df = gen_result_df(forecasters, data_desc)
    plot_metric(forecasters, result_df)
    plt.close()