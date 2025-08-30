from statsmodels.tsa.api import ExponentialSmoothing
import numpy as np
import matplotlib.pyplot as plt


class Forecaster_ExponentialSmoothing:
    def Forecast(self, trace, forecast_window, int_val_prediction, params):
        '''
        trace: observed time series
        n: number of values to be forecasted
        '''
        trace=np.array(trace)
        fit = ExponentialSmoothing(trace, initialization_method = "estimated").fit(optimized=True)
        # print(fit.model.params)
        # with mod.fix_params({"smoothing_level": 0.2}):
        #     fit = mod.fit(optimized=True)
        #     predictions = fit.forecast(forecast_window)
        #     print(predictions)
        predictions = fit.forecast(forecast_window)
        if int_val_prediction is True:
            quantized_predictions = [round(p) for p in predictions]
            return list(quantized_predictions), fit.params['smoothing_level']
        else:
            return list(predictions), fit.params['smoothing_level']

    def simulation(self, real_values, forecast_window):
        for j in range(len(real_values)-forecast_window):
            fcast, param = self.Forecast(trace[j:j+forecast_window], forecast_window, True, {})
            print(fcast, param)

if __name__ == "__main__":
    fes = Forecaster_ExponentialSmoothing()
    trace = [100-i for i in range(100)]
    # trace = [0,0,0,0,2,2,2,2]*20
    # print(fes.Forecast(trace, 60, True, {}))
    fes.simulation(trace, 60)

