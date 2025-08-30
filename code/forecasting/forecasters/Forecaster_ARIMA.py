import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", UserWarning)
import pmdarima as pm

# ARIMA: Autoregressive Integrated Moving Average
# more on auto arima here: 
# http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html#pmdarima.arima.auto_arima

class Forecaster_ARIMA:
    def __init__(self) -> None:
        self.model = None
    def Forecast(self, trace, forecast_window, int_val_prediction=True, params=None):
        '''
        trace: observed time series
        n: number of values to be forecasted
        '''
        if trace[-1] == 0:
            trace[-1] = 0.0001
        
        self.model = pm.auto_arima(trace, suppress_warnings=True) 
        predictions = self.model.predict(forecast_window, suppress_warnings=True)
        
        if (int_val_prediction is True):
            predictions = [int(round(max(p, 0))) for p in predictions]
        return predictions

