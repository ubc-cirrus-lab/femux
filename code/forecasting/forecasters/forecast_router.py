from forecasters.Forecaster_AR import Forecaster_AR
#from forecasters.Forecaster_SETAR import Forecaster_SETAR
from forecasters.Forecaster_ARIMA import Forecaster_ARIMA
from forecasters.Forecaster_ExponentialSmoothing import Forecaster_ExponentialSmoothing
from forecasters.Forecaster_Holt import Forecaster_Holt
from forecasters.Forecaster_FFT import Forecaster_FFT
from forecasters.Forecaster_Theta import Forecaster_Theta


class ForecastRouter:
    def __init__(self, int_val_prediction=False) -> None:
        self.current_trace_type = None
        self.forecaster_MCMC = None
        self.int_val_prediction = int_val_prediction  # by default predictions are for integer values, can be toggled

    def route_forecast(self, trace, forecast_window, forecast_method, params):
        """
        trace: observed time series (for instance, dpc trace)
        forecast_window: number of values to be forecasted
        """
        if len(trace) == 0:
            # no predictions if the input trace is empty
            return []
        if min(trace) == max(trace):
            # return same values for a constant trace
            predictions = forecast_window * [trace[0]]
            return predictions
        forecaster = None
        if forecast_method == "ARIMA":
            forecaster = Forecaster_ARIMA()
            predictions = forecaster.Forecast(
                trace, forecast_window, self.int_val_prediction, params
            )
            return predictions
        if forecast_method == "ExponentialSmoothing":
            forecaster = Forecaster_ExponentialSmoothing()
            predictions = forecaster.Forecast(
                trace, forecast_window, self.int_val_prediction
            )
            return predictions
        elif forecast_method == "AR":
            forecaster = Forecaster_AR()
            predictions = forecaster.Forecast(
                trace, forecast_window, self.int_val_prediction, params
            )
            return predictions
        elif forecast_method == "Holt":
            forecaster = Forecaster_Holt()
            predictions = forecaster.Forecast(
                trace, forecast_window, self.int_val_prediction, params
            )
            return predictions
        elif forecast_method == "Polynomial":
            forecaster = Forecaster_Polynomial()
            predictions = forecaster.Forecast(
                trace, forecast_window, self.int_val_prediction, params
            )
            return predictions
        elif forecast_method == "Theta":
            forecaster = Forecaster_Theta()
            predictions = forecaster.Forecast(
                trace, forecast_window, self.int_val_prediction, params
            )
            return predictions
        elif forecast_method == "FFT":
            forecaster = Forecaster_FFT()
            predictions = forecaster.Forecast(
                trace, forecast_window, self.int_val_prediction, params
            )
            return predictions
        elif forecast_method == "Theta_FFT_Bagging":
            forecaster_theta = Forecaster_Theta()
            forecaster_fft = Forecaster_FFT()
            predictions_theta = forecaster_theta.Forecast(
                trace, forecast_window, self.int_val_prediction, params
            )
            predictions_fft = forecaster_fft.Forecast(
                trace, forecast_window, self.int_val_prediction, params
            )
            predictions = predictions_theta
            for i in range(len(predictions)):
                # Geometric Bagging
                predictions[i] = round((max(0,predictions[i])*max(0,predictions_fft[i]))**0.5)
            return predictions
        elif forecast_method == "MCMC":
            if self.current_trace_type == None or params["trace_type"] != self.current_trace_type:
                 self.current_trace_type = params["trace_type"]
                 self.forecaster_MCMC = Forecaster_MCMC()
            (
                forecasted_values,
                across_mcmc_run_overall_error,
                mcmc_elapsed_time,
                mcmc_start_index,
            ) = self.forecaster_MCMC.Forecast(
                idpc=trace, forecast_window=forecast_window, index=params["index"]
            )
            specific_results = [
                across_mcmc_run_overall_error,
                mcmc_elapsed_time,
                mcmc_start_index,
            ]
            return forecasted_values
        else:
            predictions = forecast_window * [trace[-1]]
            return predictions

    def Setint_val_prediction(self, int_val_prediction):
        self.int_val_prediction = int_val_prediction
