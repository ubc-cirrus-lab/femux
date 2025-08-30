import numpy as np
import sys
import matplotlib.pyplot as plt
from data_processing import gen_result_df
from metric_calc import calc_objective_function

from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data") + "/"
save_path= str(Path(__file__).parents[1] / "output_plots") + "/"

MIN_METRIC = 1e-6

def plot_obj_func_winners(forecasters, result_df):
    args = dict()
    args["y_label"] = "Number of Blocks"
    args["x_label"] = "Forecaster"
    args["file_name"] = save_path + "sec_5_block_winners.pdf"

    result_df["ObjectiveWinners"] = result_df.apply(lambda x : gen_winner_list(x.NumColdStarts, x.MemoryUsed, 
                                                                               x.MemAllocated, forecasters), axis=1)

    forecaster_wins = list(np.concatenate(result_df.ObjectiveWinners.to_list()))
    forecaster_vals = []

    for forecaster in forecasters:
        num_wins = sum([1 for forecaster_win in forecaster_wins if forecaster_win[0] == forecaster])
        forecaster_vals.append(num_wins)

    plot(args, forecaster_vals, forecasters)


def gen_winner_list(num_cold_starts, mem_used, mem_alloc, forecasters):
    num_blocks = len(num_cold_starts[forecasters[0]])
    winner_list = []


    for block_index in range(num_blocks - 1):
        best_result = calc_objective_function(forecasters[0], block_index, 
                                              num_cold_starts, mem_used, mem_alloc)
        best_forecaster = forecasters[0]
        results = [best_result]

        for forecaster in forecasters[1:]:
            cur_result = calc_objective_function(forecaster, block_index, 
                                                 num_cold_starts, mem_used, mem_alloc)

            if cur_result < best_result:
                best_result = cur_result
                best_forecaster = forecaster

            results.append(cur_result)

        # flag this block as a skip if all of the forecasters are 0 (indicating nothing happened) 
        improvement_over_median = np.percentile(results, 50) - best_result
        winning_forecaster = best_forecaster if improvement_over_median > MIN_METRIC else "Skip"
        winner_list.append([winning_forecaster])

    return winner_list


def plot(args, data, forecasters):
    plt.bar(forecasters, data)
    plt.xlabel(args["x_label"])
    plt.ylabel(args["y_label"])

    plt.savefig(args['file_name'], bbox_inches="tight")

if __name__ == "__main__":
    forecasters = ["ExpSmoothing", "MarkovChain", "Holt", "AR", "FFT_10", "10_min_keepalive", "SETAR"]
    
    block_size = 504
    percentage = 100
    data_split = "train"
    data_desc = "{}_{}_percent_{}".format(block_size, percentage, data_split)

    result_df = gen_result_df(forecasters, data_desc)

    plot_obj_func_winners(forecasters, result_df)

    plt.close()