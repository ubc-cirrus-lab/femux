import pandas as pd
import numpy as np
from plotter import plot
from pathlib import Path

data_path = str(Path(__file__).parents[2] / "azure" / "results" / "mae" ) + "/"
plot_path = str(Path(__file__).parents[1] / "output_plots") + "/"

NUM_CHUNKS = 40
NUM_MED_CHUNKS = 4
NUM_LARGE_CHUNKS = 1

conc_forecast_save_path = "../../data/azure/forecaster_data/concurrency/conc_forecast_00.pickle"
stats_save_path = "../../data/azure/plotter_data/small_mae_cdf_data.pickle"

mae_save_path = data_path + "{}_maes.pickle"


os.makedirs()

def set_paths(data, filter_mode, num_steps):
    # For Azure data there's more work to do
    data_path = conc_forecast_save_path.replace("conc_", "conc_{}_".format(filter_mode))

    save_path = stats_save_path.replace("small_", "{}_{}_{}_step".format(filter_mode, mode, num_steps))
    num_chunks = NUM_CHUNKS

    if filter_mode == "medium":
        num_chunks = NUM_MED_CHUNKS
    elif filter_mode == "large":
        num_chunks = NUM_LARGE_CHUNKS

        
    return data_path, save_path, num_chunks


def gather_vals(forecasters, data, stats, mode, filter_mode, num_steps, use_cache=False):
    mae_df = pd.DataFrame(columns=stats, index=forecasters)
    data_path, save_path, num_chunks = set_paths(data, mode, filter_mode, num_steps)
    
    if use_cache:
        mae_df = pd.read_pickle(save_path)
    
    for forecaster in forecasters:
        print("parsing forecaster {}".format(forecaster))
        forecast_path = data_path.replace("_forecast", "_forecast_" + forecaster)

        ninety_nine_percentile_vals = []
        median_vals = []
        avg_vals = []

        for chunk_index in range(num_chunks):
            print("on chunk {}".format(chunk_index))
            forecaster_df = pd.read_pickle(forecast_path.replace("_00", "_{:02d}".format(chunk_index)))
            forecaster_df["MAE"] = forecaster_df.MAE.apply(lambda x : get_mae_for_num_steps(x, num_steps))
            forecaster_df.sort_values(by=["HashApp"], inplace=True)
            forecaster_df = forecaster_df.drop_duplicates(subset='HashApp', keep='first').reset_index(drop=True)

            mae_lists = forecaster_df.MAE.values.tolist()

            for mae_list in mae_lists:
                if len(mae_list) > 0:
                    # we remove first block since forecasting starts midway through
                    mae_list = mae_list[383:]
                    blocks = np.split(mae_list, 39)
                    
                    for block in blocks:
                        ninety_nine_percentile_vals.append(np.percentile(block, 99))
                        median_vals.append(np.percentile(block, 50))
                        avg_vals.append(np.average(block))

        mae_df.at[forecaster, "Median"] = median_vals
        mae_df.at[forecaster,"99thPercentile"] = ninety_nine_percentile_vals
        mae_df.at[forecaster, "Avg"] = avg_vals

    mae_df.to_pickle(save_path)


def get_maes_per_block(data_df, stat): 
    maes_by_forecaster = data_df[stat].to_list()
    
    # build a list where each sub-list contains the average mae for each forecaster for the 
    # given block
    forecaster_maes_by_block = []
    num_traces = len(maes_by_forecaster[0])

    for i in range(num_traces):
        forecaster_maes_by_block.append([block_mae[i] for block_mae in maes_by_forecaster])

    return forecaster_maes_by_block


def get_mae_for_num_steps(mae_list, num_steps):
    if type(mae_list[0]) != np.ndarray:
        if num_steps > 1:
            raise ValueError("Can't do multistep MAE if there is only one step")
        else: 
            return mae_list
    
    return [mae[num_steps - 1] for mae in mae_list]


def plot_mae_cdfs(forecasters, data, stat, mode, filter_mode, num_steps):
    """Plot MAEs for forecasters for the given number of steps (1 for single step
    forecasting, or the 2nd, 3rd,... step for multi-step)

    forecasters: list[str]
    Names of forecasters to plot

    data: list[float]
    MAE list

    stat: str
    Which statistic to show for the trace: Average, 99th p, Median

    filter_mode: str
    Size of applications: small (<1M invocations), medium (1M-100M), large (>100M)

    num_steps: int
    number of steps in the future for each forecast. If this value is 5, that means
    at each timestep we have the MAE of the forecast 5 timesteps in the future.
    """

    cdf_args = dict()

    _, save_path, _ = set_paths(data, mode, filter_mode, num_steps)
    mae_df = pd.read_pickle(save_path)

    cdf_args["MAE_Stat"] = stat
    cdf_args["FilterMode"] = filter_mode
    cdf_args["Mode"] = mode
    cdf_args["NumSteps"] = num_steps
    cdf_args["Extra"] = ""
    cdf_args["Data"] = data
    
    close = False
    for i, forecaster in enumerate(forecasters):
        if i == len(forecasters) - 1:
            close = True

        cdf_args["Forecaster"] = forecaster
        vals = mae_df.loc[forecaster][stat]

        plot(vals, 99, cdf_args, close)

def plot_mae_boxplot(forecasters, data, stat, mode, filter_mode, num_steps):
    """Plot MAEs for forecasters for the given number of steps (1 for single step
    forecasting, or the 2nd, 3rd,... step for multi-step)

    forecasters: list[str]
    Names of forecasters to plot

    data: list[float]
    MAE list

    stat: str
    Which statistic to show for the trace: Average, 99th p, Median

    filter_mode: str
    Size of applications: small (<1M invocations), medium (1M-100M), large (>100M)

    num_steps: int
    number of steps in the future for each forecast. If this value is 5, that means
    at each timestep we have the MAE of the forecast 5 timesteps in the future.
    """

    plot_args = dict()

    _, save_path, _ = set_paths(data, mode, filter_mode, num_steps)
    mae_df = pd.read_pickle(save_path)
    mae_df = mae_df.loc[forecasters]

    forecaster_wins = gen_winning_forecast_dict(mae_df, forecasters, stat)    

    plot_args["label"] = "Forecasters"
    plot_args["y_label"] = "{} MAE".format(stat)
    plot_args["x_label"] = "Forecaster"
    plot_args["file_name"] = plot_path + "{}/forecaster_perf/{}/{}_{}_step_mae_{}_pdf".format(data, 
                            filter_mode, mode, num_steps, stat) 
    plot_args["mae_boxplot"] = True

    forecaster_vals = []
    plot_args["forecaster_labels"] = []

    for forecaster in forecasters:
        forecaster_vals.append(forecaster_wins[forecaster])
        plot_args["forecaster_labels"].append("{} ({})".format(forecaster, len(forecaster_wins[forecaster])))

    plot(forecaster_vals, 99, plot_args, close=True)


def gen_winning_forecast_dict(mae_df, forecasters, stat):
    """
    mae_df: pd.Dataframe()
        index: forecaster name
        Avg: list[float]
            average MAE per trace, order is maintained across all forecasters
    
    returns: dict(str, list[float])
    key is forecaster name and value is a list of MAEs for when the forecaster wins.
    """
    
    forecaster_maes_by_block = get_maes_per_block(mae_df, stat)
    winning_forecasters = [maes.index(min(maes)) for maes in forecaster_maes_by_block]
    forecaster_indices = [list(mae_df.index).index(forecaster) for forecaster in forecasters]

    forecaster_wins = dict()
    for i, forecaster in enumerate(forecasters):
        forecaster_index = forecaster_indices[i]
        forecaster_vals = []

        for block_index, winning_forecaster in enumerate(winning_forecasters):
            if winning_forecaster == forecaster_index:
                forecaster_vals.append(forecaster_maes_by_block[block_index][winning_forecaster])
        
        forecaster_wins[forecaster] = forecaster_vals

    return forecaster_wins


if __name__ == "__main__":
    stats = ["Median", "99thPercentile", "Avg"]
    forecasters = ["AR", "ExpSmoothing", "FFT", "Holt", "SETAR", "MarkovChain"]
    data = "azure"
    num_steps = 1

    for size in ["small", "medium", "large"]:
        gather_vals(forecasters, data, stats, "concurrency", size, num_steps, use_cache=False)
        for stat in stats:
            #plot_mae_cdfs(forecasters, data, stat, "concurrency", size, num_steps)
