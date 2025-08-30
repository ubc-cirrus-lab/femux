import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as mticker
import sys
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler, MinMaxScaler
from pathlib import Path

save_path = str(Path(__file__).parents[1] / "output_plots") + "/"
markers = ["d", "v", "s", "*", "^", "d", "v", "s", "*", "^", "d", "v", "s", "*", "^", "d", "v", "s", "*", "^"]
plt.rc('font', size=12)

"""
args = {"block_size": <block_size>}
"""

def plot(data, bins, args, close=False):
    if "forecaster" in args or "ColName" in args:
        count, bins_count = np.histogram(data, bins=bins)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)

    if "ColName" in args:
        if args["log"] == True:
            data = [np.round(val, 5) for val in data]
            cdfx = np.sort(data)
            cdfy = np.linspace(1 / len(data), 1.0, len(data))
            plt.plot(cdfx, cdfy, label=args["ColName"])
            plt.xscale("log")
        else:
            plt.plot([bc*100 for bc in bins_count[1:]], cdf, label="{}".format(args["ColName"]))
            plt.xlabel("")

        plt.ylabel("Cumulative Fraction of {}".format(args["yAxisTitle"]))
        plt.legend(title = "Features")
        plt.savefig(save_path + "{}_{}_cdf.pdf".format(args["ColName"], args["Title"]))

    elif "barplot" in args:
        plt.bar(args["x"], args["y"])
        plt.xlabel(args["x_label"])
        if "y_label" in args:
            plt.ylabel(args["y_label"])
        plt.savefig(args['file_name']+'.pdf')
    
    elif "MAE_Stat" in args:
        data = [round(val, 5) for val in data]
        cdfx = np.sort(data)
        cdfy = np.linspace(1 / len(data), 1.0, len(data))
        plt.plot(cdfx, cdfy, label=args["Forecaster"])
        plt.xlabel(args["MAE_Stat"] + " MAEs of Blocks")
        plt.xscale("log")
        plt.ylabel("Cumulative Fraction of Applications")
        plt.legend(title = "Forecasters")
        plt.savefig(save_path + "{}/forecaster_perf/{}/{}_{}_step_mae_{}_{}_cdf.pdf".format(args["Data"], 
        args["FilterMode"], args["Mode"], args["NumSteps"], args["MAE_Stat"], args["Extra"]))
    elif "forecaster" in args:
        plt.title(args['title'])
        plt.plot([bc*100 for bc in bins_count[1:]], cdf, label="{}".format(args["forecaster"]))
        plt.xlabel(args['x_label'])
        plt.ylabel(args['y_label'])
        plt.legend(title = args['legend_title'])
        plt.savefig(args['file_name']+'.pdf')
    elif "RMSE" in args:
        plt.title(args['title'])
        data = [round(val, 5) for val in data]
        cdfx = np.sort(data)
        cdfy = np.linspace(1 / len(data), 1.0, len(data))
        if 'dashed' in args:
            plt.plot(cdfx, cdfy, label=args['label'], linestyle='dashed')
        else:
            plt.plot(cdfx, cdfy, label=args['label'])
        plt.xlabel(args["x_label"])
        plt.xscale("log")
        plt.ylabel(args['y_label'])
        plt.legend(prop={'size': 6})
        plt.savefig(args['file_name']+'.pdf')
    elif "x" in args:
        # scatter plot       
        plt.set_cmap("cividis")
        data = [round(val, 6) for val in data]
        if "log" in args and args["log"]:
            plt.yscale("log")

        marker = markers[args["num"]] if "num" in args else "o"
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.scatter(int(args["x"]), int(args["y"]), 80, label=args['label'], 
                    marker=marker, zorder=1)

        #if args["line"]:
            ## pareto frontier line crossing best point        
            #plt.ylim([0, 10000000])#3 * args["y_best"]])
            #plt.xlim([0, 600000])#2 * args["x_best"]])
            #if close:
                #axes = plt.gca()
                ##axes.set_aspect('equal', adjustable='box')
                #plt.gca().get_xaxis().get_offset_text().set_position((1,0))

                #min_x = axes.get_xlim()[0]
                #min_y = axes.get_ylim()[0]

                ## on the y-axis of graph
                #x1, y1 = min_x, (args["x_best"] - min_x) * 100 + args["y_best"]
            
                ## on x-axis of graph
                #x2, y2 = (args["y_best"] - min_y) / 100 + args["x_best"], min_y

                #plt.plot([x1,x2], [y1,y2], zorder=0)
                

        plt.xlabel(args["x_label"]) 
        plt.ylabel(args['y_label'])
        plt.legend()#title="Features")
        for axis in ["x", "y"]:
            if "%" in args["{}_label".format(axis)]:
                plt.ticklabel_format(style="plain")
            else:
                plt.ticklabel_format(style='sci', axis=axis, scilimits=(0,0))
        #plt.tight_layout()
        plt.savefig(args['file_name']+'.pdf', bbox_inches="tight")
    elif "line" in args:
        # line plot
        plt.title(args['title'])
        data = [round(val, 6) for val in data]
        plt.yscale("log")
        plt.plot(data, label=args["Forecaster"])
        plt.xlabel(args["x_label"]) 
        plt.ylabel(args['y_label'])
        plt.legend()
        plt.savefig(args['file_name']+'.pdf', bbox_inches="tight")
    
    elif "kmeans_box" in args:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlabel(args["x_label"]) 
        
        labels = args["y_label"]
        labels = [label.replace("_Metrics", "") for label in labels]
        ax.set_yticklabels(labels, fontsize=7)
        
        bp = ax.boxplot(data, vert = False, whis=[1,99])
        plt.title(args["title"])
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.tight_layout()
        plt.savefig(args["file_name"] +'.pdf') 
    
    elif "boxplot" in args:
        plt.boxplot(data, whis=[1, 99])
        plt.yscale("log")
        plt.xticks(list(range(1, len(data) + 1)), args["forecaster_labels"], rotation="vertical")
        plt.title(args["title"])
        plt.xlabel(args["x_label"]) 
        plt.ylabel(args['y_label'])
        plt.legend()
        plt.savefig(args['file_name']+'.pdf', bbox_inches="tight")

    else:
        plt.title(args['title'].title())
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
        plt.savefig(args['file_name']+'.pdf')
    
    if close:
        plt.close()


def hist_plot(data, bins, args):
    plt.hist(data, bins=bins, label="{}".format(args["ColName"]))
    plt.xlabel("")
    if args["log"]:
        plt.yscale("log")
    plt.ylabel("Number of {}".format("Blocks"))
    plt.legend(title = "Features")
    plt.savefig(save_path + "{}/{}_percent_data/{}_hist_{}_{}.pdf".format(args["ColName"], args["DataFraction"], 
                                                    args["Transform"], bins, args["log"]))
    plt.close()


def transformer_hist(filepath, data_fraction, column_names, transform="", bins=99, log=False):
    df = pd.read_pickle(filepath)
    
    if transform:
        df = df.explode(column_names)
        df = df[column_names]
        scaler = choose_transform(transform)
        df = scaler.fit_transform(df)

    args = dict()
    args["log"] = log
    args["Transform"] = transform
    args["DataFraction"] = data_fraction

    for i, column_name in enumerate(column_names):
        args["ColName"] = column_name

        if transform:
            joined_vals = [val[i] for val in df]
        else:
            vals = df[column_name].values.tolist()
            joined_vals = list(np.concatenate(vals).flat) 

        hist_plot(joined_vals, bins, args)

def choose_transform(transformer):
    if transformer == "QuantileNormal":
        return QuantileTransformer(output_distribution="normal")
    elif transformer == "QuantileUniform":
        return QuantileTransformer()
    elif transformer == "PowerTransformer":
        return PowerTransformer()
    elif transformer == "StandardScaler":
        return StandardScaler()
    elif transformer == "MinMaxScaler":
        return MinMaxScaler()
    else:
        print("not found")


if __name__ == '__main__':
    
    data_file = "../../data/big_data/representative_traces_70_percent_top_n_extraction.pickle"
    data_percentage = 70

    column_names = ["EMADensity", "EMAStationarity", "EMALinearity"]
    transformers = ["QuantileNormal", "QuantileUniform", "PowerTransformer", "StandardScaler", "MinMaxScaler"]