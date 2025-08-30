import concurrent
import numpy as np
import pandas as pd
import sys
from math import isclose
from time import time
from post_processing import get_combined_transformed_and_forecasted_values, convert_it_forecast_to_avg_conc
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
from collections import deque
from sim_interface import SimulatorInterface
from forecasters import setar_model
#from darts.models import FFT
#from darts import TimeSeries
from forecasters.Forecaster_MarkovChain import Forecaster_MarkovChain
from markov_chain import MarkovChainRadical
from forecasters.Forecaster_IceBreaker import fourierExtrapolation
from tuned_exp_smoothing import TunedForecaster

sys.path.append("..")
from forecasting.forecasting_sim_utils import gen_obj_val
from results.utils import add_mem_values, add_exec_times

SETAR_MIN_REGIME_FRAC = 0.1
SOFT_SCALING_LIMIT = 3000
SCALING_INCREMENT = 500
MINUTES_PER_DAY = 1440
SECONDS_PER_MIN = 60
MS_TO_SECOND = 1000
MB_to_GB = 1000
STATIC_KEEPALIVE_WINDOW = 10


# referenced for ExpSmoothing/Holt https://stats.stackexchange.com/questions/475749/how-to-update-an-exponentialsmoothing-model-on-new-data-without-refitting

class ForecastSimulation(SimulatorInterface):
    def __init__(self, forecaster, weight_mode="default", param=None, forecast_len=1, num_past_elements=120, block_size=504, func_mode = False, data_mode="concurrency"):
        self.forecast_len = forecast_len
        self.forecaster = forecaster
        self.num_past_elements = num_past_elements
        self.block_size = block_size
        self.func_mode = func_mode
        self.default_param = param
        self.weight_mode = weight_mode
        self.idletime_mode = data_mode == "idletime"
        self.ades_alpha = 0.5
        self.ades_beta = 0.5

        print(num_past_elements)

        print("Initializing forecasting simulator with {} and parameter {} for {} forecasts".format(forecaster, param, data_mode))
        # flag is set when post processing is required in case of partitioning after func level trace processing


    def run_sim(self, traces_df, num_workers):
        """Run forecasting simulation with multiprocessing capabilities.
            traces_df: dataframe
            contains "TransformedValues".
                TransformedValues: list[float or int]
                time series to get forecasted values

            num_workers: int
            number of cores to use

            returns: dataframe
                TransformedValues
                ForecastedVals: np.arr(np.arr(float))
                the forecasted value(s) at each time step
        """
        traces_df.reset_index(inplace=True, drop=True)

        traces_df = add_mem_values(traces_df)
        traces_df = add_exec_times(traces_df)

        # some apps don't have memory values
        traces_df = traces_df.dropna()
        traces_df.drop(traces_df[traces_df["HashApp"] == "cb34fd874e255ddeaf1a38d86e7f3a41dffb0efdee8e46329d4ba8c2dad0fab5"].index, inplace=True)
        traces_df.reset_index(drop=True, inplace=True)

        split_traces_df = np.array_split(traces_df, num_workers)

        with concurrent.futures.ProcessPoolExecutor(max_workers = num_workers) as executor:
            results = executor.map(self._run_forecast_sim, split_traces_df)
 
        result = pd.concat(results)
        result.reset_index(drop=True, inplace=True)
        
        if self.func_mode:
            result = get_combined_transformed_and_forecasted_values(result, self.forecast_len)

        result.drop("TransformedValues", axis=1, inplace=True)

        return result


    def _run_forecast_sim(self, traces_df):
        """Generate the MAEs for each forecaster for each trace.

        traces_df: dataframe
        contains "TransformedValues".
            TransformedValues: list[float or int]
            time series to get forecasts for

        returns: dataframe
        adds a "Forecaster" and "MAE" column
            MAE: list[float]
            Mean Absolute Error for each observation

            Forecaster: str
            Name of forecaster used
        """
        if traces_df.shape[0] == 0:
            return traces_df

        if self.func_mode and self.idletime_mode:
            raise Exception("Idletime not supported for func-level forecasts")
            
        if self.func_mode:
            # get each application
            fcastmat = []
            app_level_transformed_values = []
            for curr_app in traces_df['HashApp'].unique():
                curr_df = traces_df[traces_df['HashApp'] == curr_app]
                per_app_transformed_values = [0]*len(curr_df.at[curr_df.index.tolist()[0], 'TransformedValues'])
                for _, row in curr_df.iterrows():
                    for i in range(len(row.TransformedValues)):
                        per_app_transformed_values[i] = max(per_app_transformed_values[i], row.TransformedValues[i])
                app_level_transformed_values.append(per_app_transformed_values)
            for curr_app in traces_df['HashApp'].unique():
                app_df = traces_df[traces_df['HashApp'] == curr_app]
                forecasted_matrix = []   
                # iterate over each function inside an app
                for _, row in app_df.iterrows():
                    # perform forecasting across all functions inside the application
                    forecasted_mat = self.forecast_trace(row.TransformedValues,
                                        self.forecaster, row.AverageMemUsage, row.ContainerInvocationsPerMin,
                                        row.ExecDurations)
                    # append both forecasted values at each index and future values at each index per function
                    forecasted_matrix.append(forecasted_mat)
                
                app_forecasted_values = []
                # outer loop for iterating over the number of times values are forecasted
                for i in range(len(forecasted_matrix[0])):
                    local_app_forecasted_values = []
                    # inner loop for iterating over the forecast horizon
                    for j in range(len(forecasted_matrix[0][0])):
                        max_fcast_per_col = 0
                        # inner most loop for iterating over all functions inside the app
                        for k in range(len(forecasted_matrix)):
                            # max value for a single function, at an index of the forecast horizon (e.g. 5)
                            # and for an index among the total number of times forecast has happend e.g. 20035
                            max_fcast_per_col = max(max_fcast_per_col, forecasted_matrix[k][i][j])
                        local_app_forecasted_values.append(max_fcast_per_col)
                    app_forecasted_values.append(local_app_forecasted_values)
                fcastmat.append(app_forecasted_values)
            all_apps = [app_name for app_name in traces_df['HashApp'].unique()]
            new_df = pd.DataFrame()
            new_df['HashApp'] = all_apps
            new_df['ForecastedValues'] = fcastmat
            new_df['ForecastedValues'] = new_df.ForecastedValues.apply(lambda x : np.array(x))
            new_df['TransformedValues'] = app_level_transformed_values
            new_df['Forecaster'] = self.forecaster
            new_df.reset_index(drop=True, inplace=True)
            return new_df
          
        traces_df["ForecastedValues"] = traces_df.apply(lambda x : self.forecast_trace(x.TransformedValues,
                                                            self.forecaster, x.AverageMemUsage, 
                                                            x.ContainerInvocationsPerMin,
                                                            x.ExecDurations, x.HashApp), axis=1)

        traces_df["Forecaster"] = self.forecaster

        return traces_df
    
    
    def forecast_trace(self, trace, forecaster, mem_vals, invocations, exec_times):
        """At every timestep in the trace, forecast self.forecast_len future elements 

        trace: list[]
        function trace represented by a time series (e.g., concurrency per second)

        forecaster: str
        forecasting model to use

        returns: np.array(np.array(float or int))
        list containing the forecasted value(s) at each timestep
        """
        trace_len = len(trace)
        forecast_window_len = self.forecast_len

        if trace_len < (self.num_past_elements + forecast_window_len):
            if self.idletime_mode:
                return np.array([[idletime] for idletime in trace])
            else:
                return np.empty(0)

        # Last two days are missing mem values so we use the average
        # across first 12 days
        mem_vals = [mem_val / MB_to_GB for mem_val in mem_vals]
        mem_vals.extend([np.average(mem_vals), np.average(mem_vals)])
        mem_usage = mem_vals[0]
        cur_day = 0

        # execution time data for execution-aware forecasters
        if self.weight_mode == "exec":
            exec_times = [exec_time / MS_TO_SECOND for exec_time in exec_times]
        else:
            exec_times = [1 for exec_time in exec_times]
        
        exec_time = exec_times[0]
        stop_index = trace_len - self.forecast_len

        # for idle times repeat the previous element until there are enough observations to start
        # forecasting. Naturally, one extra idletime is forecasted since the forecaster doesn't know
        # when the trace ends.
        if self.idletime_mode:
            forecast_list = np.zeros((stop_index, self.forecast_len))
            forecast_list[:self.num_past_elements - 1] = np.array([[idletime] for idletime in trace[:self.num_past_elements - 1]])
            forecaster_offset = 0
            start_index = self.num_past_elements - 1
        else:
            # minimum number of past elements to perform forecasting
            forecast_list = np.zeros((stop_index - self.num_past_elements, self.forecast_len))
            start_index = self.num_past_elements
            forecaster_offset = self.num_past_elements

        # State variables 
        observed_vals = deque(trace[:self.num_past_elements])
        forecasted_values = np.empty(self.forecast_len)
        repeat_elements = self._check_repeats(trace[:self.num_past_elements])    
        num_forecasts = 0
        min_regime_num = np.ceil(0.1 * self.num_past_elements)
        num_thresholds = 1

        # adaptive
        if "Adaptive" in forecaster:
            cur_param = self.default_param
            params, perfs, num_params, cur_param = self._set_params(forecaster)
            if forecaster != "Adaptive_Double_Exponential_Smoothing":
                cur_param = self.default_param

        if "MarkovChain" in forecaster:
            mc_model = MarkovChainRadical(trainingSet=trace[:self.num_past_elements])
        else:
            mc_model = None

        for cur_element in range(start_index, stop_index):
            
            if cur_element % MINUTES_PER_DAY == 0:
                cur_day += 1
                mem_usage = mem_vals[cur_day]
                exec_time = exec_times[cur_day]

            # restart forecasting cadence at start of each block.
            if cur_element % self.block_size == 0:
                num_forecasts = 0

            trace_val = trace[cur_element] 

            if "MarkovChain" in forecaster:
                forecasted_values = mc_model.Forecast(list(observed_vals), self.forecast_len)
            
            elif repeat_elements < self.num_past_elements:
                # If switching threshold then retrain
                if forecaster == "SETAR":
                    observations = np.array(observed_vals)
                    num_possible_thresholds = len(np.unique(observations[:-self.default_param])) / min_regime_num
                    new_num_thresholds = 2 if num_possible_thresholds > 3 else 1
                                        
                    if new_num_thresholds != num_thresholds:
                        num_forecasts = 0

                    num_thresholds = new_num_thresholds

                
                if forecaster == "AR":
                    mod = AutoReg(list(observed_vals), lags=self.default_param, trend='c', seasonal=False).fit()
                    forecasted_values = mod.predict(start=self.num_past_elements, 
                                                    end=self.num_past_elements + self.forecast_len - 1)
                   
                elif forecaster == "SETAR":
                    try:
                        if num_forecasts % 30 == 0 or num_forecasts < 5:
                            mod = setar_model.SETAR(observations, order=num_thresholds, 
                                                    ar_order=self.default_param).fit()

                        forecasted_values = mod.forecast(steps=forecast_window_len, initial=observations)
                    except Exception:
                        # use default forecaster if there is a singular matrix

                        mod = AutoReg(list(observed_vals), lags=self.default_param, trend='c', seasonal=False).fit()
                        forecasted_values = mod.predict(start=self.num_past_elements, 
                                                        end=self.num_past_elements + self.forecast_len - 1)

                elif forecaster == "FFT":
                    # Parameters and rounding taken from IceBreaker source code. Making 
                    # negative values 0 is done in _growth_cap()
                    forecasted_values = fourierExtrapolation(list(observed_vals), self.forecast_len, n_harmonics=self.default_param)
                    forecasted_values = forecasted_values[self.num_past_elements:self.num_past_elements+self.forecast_len]

                elif forecaster == "Holt":    
                    if num_forecasts % 30 == 0 or num_forecasts < 5:
                        mod = ExponentialSmoothing(list(observed_vals), trend=True).fit(disp=0)
                    else:
                        mod = mod.append([observed_vals[-1]])

                    forecasted_values = mod.forecast(steps=self.forecast_len)
                    
                elif forecaster == "ExpSmoothing":
                    if num_forecasts % 30 == 0 or num_forecasts < 5:
                        mod = ExponentialSmoothing(list(observed_vals)).fit(disp=0)
                    else:
                        mod = mod.append([observed_vals[-1]])

                    forecasted_values = mod.forecast(steps=self.forecast_len)

                elif forecaster == "IceBreaker":
                    # Parameters and rounding taken from IceBreaker source code. Making negative values 0 
                    # is done in _growth_cap()
                    forecasted_values = fourierExtrapolation(list(observed_vals)[-60:], 1)
                    forecasted_values = [round(forecasted_values[len(forecasted_values) - 1])]

                elif forecaster == "Adaptive_FFT":
                    prev = time()
                    for i in range(num_params):
                        cur_forecasts = fourierExtrapolation(list(observed_vals), 1, n_harmonics=params[i])
                        cur_forecasts = cur_forecasts[self.num_past_elements:self.num_past_elements+self.forecast_len]

                        perfs[i] = gen_obj_val(self.weight_mode, cur_forecasts[0], trace_val,
                                               invocations[cur_element], mem_usage, exec_time)
                        
                        if params[i] == cur_param:
                            forecasted_values = cur_forecasts

                    cur_param = params[np.argmin(perfs)]

                elif forecaster == "Adaptive_ExpSmoothing":
                    for i in range(num_params):
                        mod = SimpleExpSmoothing(list(observed_vals)).fit(smoothing_level=params[i], 
                                                                          optimized=False)

                        cur_forecasts = mod.forecast(steps=self.forecast_len)

                        perfs[i] = gen_obj_val(self.weight_mode, cur_forecasts[0], trace_val,
                                               invocations[cur_element], mem_usage, exec_time)
                        
                        if params[i] == cur_param:
                            forecasted_values = cur_forecasts

                    cur_param = params[np.argmin(perfs)]
                elif forecaster == "Adaptive_Double_Exponential_Smoothing":
                    # forecasting with current parameters
                    alpha = cur_param['alpha']
                    beta = cur_param['beta']
                    ema1 = 0
                    ema2 = observed_vals[1] - observed_vals[0]
                    for j in range(1, len(observed_vals)):
                        oldema1 = ema1
                        ema1 = alpha * observed_vals[j] + (1 - alpha) * (ema1 + ema2)
                        ema2 = beta * (ema1 - oldema1) + (1 - beta) * ema2
                    forecasted_values = [(ema1 + ema2)]

                    # exploration
                    par = []
                    forecasts = np.zeros(2*num_params)
                    forecaster_index = 0
                    beta = cur_param['beta']
                    for alpha in params:
                        ema1 = 0
                        ema2 = observed_vals[1] - observed_vals[0]
                        for j in range(1, len(observed_vals)):
                            oldema1 = ema1
                            ema1 = alpha * observed_vals[j] + (1 - alpha) * (ema1 + ema2)
                            ema2 = beta * (ema1 - oldema1) + (1 - beta) * ema2
                        forecasts[forecaster_index] = (ema1 + ema2)
                        forecaster_index += 1
                        par.append("alpha")
                    alpha = cur_param['alpha']
                    for beta in params:
                        ema1 = 0
                        ema2 = observed_vals[1] - observed_vals[0]
                        for j in range(1, len(observed_vals)):
                            oldema1 = ema1
                            ema1 = alpha * observed_vals[j] + (1 - alpha) * (ema1 + ema2)
                            ema2 = beta * (ema1 - oldema1) + (1 - beta) * ema2
                        forecasts[forecaster_index] = (ema1 + ema2)
                        forecaster_index += 1
                        par.append("beta")

                    for i in range(len(forecasts)):
                        perfs[i] = gen_obj_val(self.weight_mode, forecasts[i], trace_val, invocations[cur_element], mem_usage, exec_time)
                    if par[np.argmin(perfs)] == "alpha":
                        cur_param['alpha'] = params[np.argmin(perfs)]
                    elif par[np.argmin(perfs)] == "beta":
                        cur_param['beta'] = params[np.argmin(perfs) - num_params]

                elif forecaster == "Adaptive_AR":
                    # get performance of all alternative params
                    for i in range(num_params):
                        mod = AutoReg(list(observed_vals), lags=params[i], trend='c', seasonal=False).fit()
                        cur_forecasts = mod.predict(start=self.num_past_elements, 
                                                        end=self.num_past_elements + self.forecast_len - 1)
                        
                        
                        perfs[i] = gen_obj_val(self.weight_mode, cur_forecasts[0], trace_val,
                                                invocations[cur_element], mem_usage, exec_time)
                        
                        if params[i] == cur_param:
                            forecasted_values = cur_forecasts 
                    
                    cur_param = params[np.argmin(perfs)]

                elif forecaster == "10_min_keepalive":
                    forecasted_values = [max(trace[cur_element - 10:cur_element])] * self.forecast_len

                elif forecaster == "5_min_keepalive":
                    forecasted_values = [max(trace[cur_element - 5:cur_element])] * self.forecast_len

                elif forecaster == "Default_Knative":
                    forecasted_values = [trace[cur_element - 1]] * self.forecast_len

                else:
                    raise Exception("No forecaster chosen")
            
                num_forecasts += 1
                        
            else:
                forecasted_values = [trace[cur_element - 1]] * self.forecast_len

                # since we stopped forecasting due to repeat elements, we restart our forecasting cadence
                num_forecasts = 0

            if not self.idletime_mode:
                forecasted_values = self._growth_cap(forecasted_values, trace_val)

            forecast_list[cur_element - forecaster_offset] = forecasted_values

            # if number is not same as last set repeat_elements to 1 since 119 repeats means that 120 numbers are the same
            repeat_elements = repeat_elements + 1 if isclose(trace[cur_element], trace[cur_element - 1],
                                                             abs_tol=0.000000001) else 1

            # slide our windows right by 1
            observed_vals.popleft()
            observed_vals.append(trace[cur_element])

        return forecast_list


    def _growth_cap(self, forecasted_values, last_observation):
        """Limit the incremental increase in forecasted values 
        """
        forecasted_values[0] = max(forecasted_values[0], 0)

        if forecasted_values[0] > SOFT_SCALING_LIMIT:    
            forecasted_values[0] = min(forecasted_values[0], 
                                max(last_observation + SCALING_INCREMENT, SOFT_SCALING_LIMIT))

        # lower bound forecasts by 0 and upper bound growth by scaling limit
        for i in range(1, self.forecast_len):
            forecasted_values[i] = max(forecasted_values[i], 0)

            if forecasted_values[i] > SOFT_SCALING_LIMIT:
                forecasted_values[i] = min(forecasted_values[i], 
                                    max(forecasted_values[i-1] + SCALING_INCREMENT, SOFT_SCALING_LIMIT))
    
        return forecasted_values


    def _check_repeats(self,trace):
        num_repeating = 1
    
        for i in range(1,len(trace)):
            if isclose(trace[i], trace[i-1], abs_tol=0.000000001):
                num_repeating += 1
            else:
                num_repeating = 1

        return num_repeating
    
    def _set_params(self, forecaster):
        perfs = np.zeros(10)
        cur_param = None
        if forecaster == "Adaptive_AR":
            params = np.arange(2, 22, 2)
        
        elif forecaster == "Adaptive_ExpSmoothing":
            params = np.arange(0, 1.05, 0.1)
            perfs = np.zeros(len(params))
        elif forecaster == "Adaptive_Double_Exponential_Smoothing":
            params = np.arange(0.05, 1.0, 0.1)
            perfs = np.zeros(2*len(params))
            cur_param = {'alpha': 0.05, 'beta': 0.05}
        elif forecaster == "Adaptive_FFT":
            params = np.arange(2, 22, 4)
            perfs = np.zeros(5)
        else:
            raise Exception("Adaptive forecaster {} not supported".format(forecaster))

        return params, perfs, len(params), cur_param
