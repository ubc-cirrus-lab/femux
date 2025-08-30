import sys

sys.path.append("./forecasters")
from markov_chain import MarkovChainRadical
import numpy as np
import time

sys.path.append("../../../../plotters")
# from superimposed_fdpc_plot import SuperImposedFdpcPlot


class Forecaster_MarkovChain:
    def Forecast(self, trace, forecast_window, params):
        """
        trace: observed time series
        forecast_window: number of values to be forecasted
        """
        predictions = params["model"].Forecast(trace, forecast_window)
        return predictions


if __name__ == "__main__":
    forecaster = Forecaster_MarkovChain()
    # trace = sum([[0]*2+[1]*2+[2]*1+[3]*2+[1]*2]*2016, [])
    # trace = [i for i in range(1,20160)]
    # trace = [0]*(20160//2)+[1]+[0]*(20158//2)
    # trace = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]*10
    # trace = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]*2
    # trace = [
    #     1.00000009,
    #     1.00032,
    #     1.5099,
    #     2.90,
    #     2,
    #     2,
    #     2,
    #     2,
    #     2,
    #     2,
    #     2,
    #     2,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    # ] * 20
    # trace = [0,5,5]*200
    trace = [i for i in range(200)]
    start_overall = time.time()
    model = MarkovChainRadical(trace[:120], ["pod_count"])
    print(
        "training complete ",
        model.state_transition_adjacency_list,
        model.state_transition_sum_list,
    )
    forecast_window = 1
    forecast_matrix = []
    flag = True
    for i in range(119, len(trace) - forecast_window - 3):
        # print("iter: ", i, trace[i:i+120])
        start = time.time()
        predictions = forecaster.Forecast(
            trace=trace[i : i + 3],  # i:i+120
            forecast_window=forecast_window,
            params={"model": model, "first_time": flag},
        )
        print("time taken for forecast: ", time.time() - start)
        forecast_matrix.append(predictions)
        # get discovered states 
        discovered_states = model.discovered_states
        # get mae values
        mae_values = abs(predictions[0] - trace[i+3])
        print(f"mae values: {mae_values} discovered_states {discovered_states}")
        flag = False
        print(f"sent value: {trace[i:i+3]} and predicted value: {predictions}")
    print(
        "this is the forecast: ",
        forecast_matrix,
        model.state_transition_adjacency_list,
        model.state_transition_sum_list,
    )
    print("total time taken: ", time.time() - start_overall)