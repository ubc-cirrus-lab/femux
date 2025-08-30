import pandas as pd
import numpy as np

preproc_path = "../data/azure/preproc_data/invocation_data/preprocessed_data_00.pickle"

func_event_path = "../data/azure/transformed_data/conc_events/func/small_func_conc_events_00.pickle"
app_event_path = "../data/azure/transformed_data/conc_events/app/medium_app_conc_events_00.pickle"
func_conc_path = "../data/azure/transformed_data/concurrency/func/small_func_conc_00.pickle"
app_conc_path = "../data/azure/transformed_data/concurrency/app/{}_app_conc_{:02d}.pickle"
invocation_path = "../data/azure/preproc_data/invocation_data/preprocessed_data_{:02d}.pickle"

forecaster_data = "../data/azure/forecaster_data/concurrency/conc_{}_forecast_{}_{:02d}.pickle"
icebreaker_data = "../data/azure/forecaster_data/invocations/IceBreaker_forecasts_{:02d}.pickle"

final_metrics = "../data/azure/results/504/MarkovChain_cold_starts_wasted_mem_0.pickle"

memory_data = "../data/azure/preproc_data/memory_data.pickle"

result_data = "../data/azure/results/504_100_percent_test/HybridHist_Inv_cold_starts_wasted_mem.pickle"
result_data_2 = "../data/azure/results/504_100_percent_test/10_min_keepalive_cold_starts_wasted_mem.pickle"

num_blocks = {"small": 40, "medium": 4, "large": 1}

def invocation_val_info(hashapp, filenum=None):
    if filenum != None:
        df = pd.read_pickle(invocation_path.format(filenum))
        df = df[df["HashApp"] == hashapp]
            
        if not df.empty:
            print("Transformed values: ")
            print("before forecasting starts {}".format(list(np.ceil(df.InvocationsPerMin.to_list()[0][:120]))))
            print("after forecasting starts {}".format(list(np.ceil(df.InvocationsPerMin.to_list()[0][120:200]))))
            return
        else:
            print("provided filenum doesn't have app")


    # we reformat each chunk sequentially, but the reformatting is done in parallel
    for filenum in range(40):

        print("on file {}".format(filenum))
        df = pd.read_pickle(invocation_path.format(filenum))
        df = df[df["HashApp"] == hashapp]
        
        if not df.empty:
            print("Transformed values: ")
            print("before forecasting starts {}".format(list(np.ceil(df.InvocationsPerMin.to_list()[0][:120]))))
            print("after forecasting starts {}".format(list(np.ceil(df.InvocationsPerMin.to_list()[0][120:200]))))
            return

    print("app not found")


def transformed_val_info(hashapp, filenum=None, size=None):
    if filenum != None:
        df = pd.read_pickle(app_conc_path.format(size, filenum))
        df = df[df["HashApp"] == hashapp]
            
        if not df.empty:
            print("Transformed values: ")
            print("before forecasting starts {}".format(list(np.ceil(df.TransformedValues.to_list()[0][:120]))))
            print("after forecasting starts {}".format(list(np.ceil(df.TransformedValues.to_list()[0][120:200]))))
            return
        else:
            print("provided filenum doesn't have app")


    for size in ["large", "medium", "small"]:
    # we reformat each chunk sequentially, but the reformatting is done in parallel
        for filenum in range(num_blocks[size]):

            print("on {} file {}".format(size, filenum))
            df = pd.read_pickle(app_conc_path.format(size, filenum))
            df = df[df["HashApp"] == hashapp]
            
            if not df.empty:
                print("Transformed values: ")
                print("before forecasting starts {}".format(list(np.ceil(df.TransformedValues.to_list()[0][:120]))))
                print("after forecasting starts {}".format(list(np.ceil(df.TransformedValues.to_list()[0][120:200]))))
                return

    print("app not found")

def forecast_info(hashapp, filenum, size, forecaster, inv_mode=False):
    start_index = 120 if forecaster == "HybridHist" else 0
    if inv_mode:
        df = pd.read_pickle(icebreaker_data.format(filenum))
        start_index = 60
        df = df[df["HashApp"] == hashapp]
        print("Forecasted Values for {}".format(forecaster))
        print(list(df.InvocationsPerMin.to_list()[0][start_index:200]))
    else:
        df = pd.read_pickle(forecaster_data.format(size, forecaster, filenum))
        df = df[df["HashApp"] == hashapp]
        print("Forecasted Values for {}".format(forecaster))
        print(list(df.ForecastedValues.to_list()[0][start_index:200]))


df = pd.read_pickle("../data/azure/results/20_20_percent_deployment/Default_Knative_cold_starts_wasted_mem.pickle")
df2 = pd.read_pickle("../data/azure/results/knative_deployment_data/workload_invocation_data_max_63_100_apps.pickle")

print(df2)
df2.NumInvocations = df2.InvocationsPerMin.apply(lambda x : sum(x[:1080]))
print(df2.NumInvocations.sum())

df.NumColdStarts = df.NumColdStarts.apply(lambda x : sum(x[:54]))
df.MemAllocated = df.MemAllocated.apply(lambda x : sum(x[:54]))
df.MemoryUsed = df.MemoryUsed.apply(lambda x : sum(x[:54]))
print(df.MemAllocated.sum())
print(df.MemoryUsed.sum())
print(df.NumColdStarts.sum())
exit()

print(df.NumColdStarts.sum())
print(df.MemAllocated.sum())
print(df.MemoryUsed.sum())


#### IceBreaker
#hashapp = "7ca324d9fc836a5d4562811c11ce3719530ee919dd1fb91bcaf71942eab8240a"
#hashapp = "5d312706e735a36c6396365f0f9ac20ce734075de98388a67770a2b07613a9b3"
#invocation_val_info(hashapp)
#forecast_info(hashapp, 19, "small", "FFT_10")
#forecast_info(hashapp, 19, "small", "IceBreaker", inv_mode=True)
    


#### SITW
#hashapp = "24ec5331ccbd4c352b675574a18a721b56ecac3c1fe0328d7e2f0042ce5a6ee6"
#hashapp = "e95c097f8961e26d9533a99dacc9baf8cea905e2080b06ed4f0760374f961e51"
#transformed_val_info(hashapp)
#forecast_info(hashapp, 3, "medium", "10_min_keepalive")
#forecast_info(hashapp, 3, "medium", "HybridHist_Inv")
#forecast_info(hashapp, 3, "medium", "HybridHist")

#df1 = pd.read_pickle(result_data)
#df2 = pd.read_pickle(result_data_2)
#df1.MemAllocated = df1.MemAllocated.apply(lambda x : sum(x))
#df1.MemoryUsed = df1.MemoryUsed.apply(lambda x : sum(x))
#df2.MemAllocated = df2.MemAllocated.apply(lambda x : sum(x))
#print(df1.MemAllocated.sum())
#print(df1.MemoryUsed.sum())
#print(df2.MemAllocated.sum())
#stitch_df = pd.merge(df1[["HashApp", "MemAllocated"]], df2[["HashApp", "MemAllocated"]], on="HashApp", suffixes=(None, "_{}".format(2)))
#stitch_df["Diff"] = stitch_df.apply(lambda x : x.MemAllocated - x.MemAllocated_2,axis=1)
#print(stitch_df.sort_values(by=["Diff"]))