import unittest
import numpy as np
import pandas as pd
from forecasting_sim import ForecastSimulation
from trace_generator import TraceGenerator
import matplotlib.pyplot as plt
import statsmodels.api as sm

class TestForecastingSim(unittest.TestCase):
    def test_AR(self):
        step_uniform = [1]
        step_uniform.extend([0] * 61)
        step_uniform.append(1)
        step_uniform.extend([0] * 60)
        print(step_uniform)

        sim = ForecastSimulation("FFT")

        result = sim.forecast_trace(step_uniform, "FFT")
        print(result)


    def test_trace(self):
        transformed_df = pd.read_pickle("../../data/transformed_data/concurrency/small_app_conc_24.pckl")

        transformed_df = transformed_df.loc[transformed_df["HashApp"] == "9b2bd0f58aef25792ae59bd5e104de15f171ca2b0f9ea71e7851bce173d58e3e"]
        print(transformed_df)

        trace = transformed_df.TransformedValues.to_list()[0]

        sim = ForecastSimulation("AR")

        result = sim.forecast_trace(trace[504*5:504*7], "AR")


    def test_SETAR(self):
        dta = sm.datasets.sunspots.load_pandas().data
        dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
        del dta["YEAR"]
        print(dta)
        dta.plot(figsize=(12,8))
        endog = np.array([j for i in dta.values for j in i])
        
        sim = ForecastSimulation("FFT")

        result = sim.forecast_trace(endog, "FFT", 0,"abcde123")
        print(result)
        plt.plot(endog)
        plt.plot(result)
        plt.savefig('fft')
        print(sum(result)/len(result))


    def test_medium(self):

        transformed_df = pd.read_pickle("../data/transformed_data/concurrency/medium_app_conc_00.pckl")

        transformed_df = transformed_df.head(1)
        
        sim = ForecastSimulation("ExpSmoothing")
        
        result = sim.run_sim(transformed_df, 1)
        
        print(result.MAE)

    
    def test_sanity(self):        
        trace_types = [
            "poisson",
            "step_uniform",
            "multiple_periods",
            "periodic_non_linear",
            "idle_burst_noise",
        ]
        
        tg = TraceGenerator()
        trace_dict = tg.generate_dpc_dict(trace_types)

        forecasters = ["AR", "SETAR", "Holt", "ExpSmoothing"]
    
        mae_df = pd.DataFrame(columns=trace_types, index=forecasters)

        time_range = list(range(60))

        for trace_type in trace_types:
            trace = trace_dict[trace_type]
            plt.plot(time_range, trace[65:125], label="actual")
            actual_vals = trace[120:125]
            
            for forecaster in forecasters:
                sim = ForecastSimulation(forecaster)

                prediction = sim.forecast_trace(trace[:180], forecaster, 0,"abcde123")

                y_vals = [None] * 55
                y_vals.extend(list(prediction))

                plt.plot(time_range, y_vals, label=forecaster)
    
                mae_df.at[forecaster, trace_type] = np.average(self.mae_stats(prediction, actual_vals, 5))
                

            plt.xlabel("Time (m)")
            plt.ylabel("Average Concurrency")
            plt.legend(title = "Forecasters")
            plt.savefig("./test_output/{}.png".format(trace_type))
            plt.clf()
        
        mae_df.to_csv("./test_output/maes.csv")




if __name__ == '__main__':
    unittest.main()