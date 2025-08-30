
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({'font.size': 14})


weight_map = {'H': 1, 'M': 534, '1': 15, '5': 393, 'E': 8, 'F': 2, 'A': 22, 'S': 22}

forecasters = ["5", "10", "holt", "mc", "es", "fft", "ar", "setar"]

df = pd.DataFrame()
for i in range(len(forecasters)):
    df = pd.concat([df, pd.read_pickle(f"../../data/knative_scalability_data/per_forecaster_data_v2/{forecasters[i]}_horizontal_scalability_data.pickle")])

# inference latency
latency_map = {}
buckets = [20, 40, 60, 80, 100]
median_latency = {}
mean_latency = {}
nn_latency = {}
median_latency_ci = {}
nn_latency_ci = {}

fig, ax1 = plt.subplots(figsize=(6, 4))

for _, row in df.iterrows():
    # for bucket in buckets:
    apps = row["FORECASTER"][0] + "_" + str(row["RPS"])
    if apps not in latency_map:
        latency_map[apps] = []
    if "1" == apps[0] or "5" == apps[0]:
        latency_map[apps].extend([0]*len(row["RESPONSE_TIMES"][-1000:]))
    else:
        latency_map[apps].extend(row["RESPONSE_TIMES"][-1000:])
for key, val in latency_map.items():
        latency_map[key] = val * weight_map[key.split("_")[0]]
        if "F" in key:
            print(len(latency_map[key]))

cur_uci_med = []
cur_lci_med = []
cur_resp_med = []

cur_uci_p95 = []
cur_lci_p95 = []
cur_resp_p95 = []
overall_median_latency = {}
overall_nn_latency = {}

for bucket in buckets:
    same_bucket_list = []
    for key, val in latency_map.items():
        if f"{bucket}" in key:
            same_bucket_list += val

    # shuffle same_bucket_list to avoid any ordering bias and again get the list
    same_bucket_list.sort()
    same_bucket_list = np.random.permutation(same_bucket_list).tolist()

    chunked_p95_times = []
    chunked_med_times = []

    chunk_size = 1000

    for i in range(0, len(same_bucket_list), chunk_size):
        chunked_p95_times.append(np.percentile(same_bucket_list[i:i+chunk_size], 99))
        chunked_med_times.append(np.median(same_bucket_list[i:i+chunk_size]))

    print(len(chunked_med_times), len(chunked_p95_times))
    median = np.mean(chunked_med_times)
    median_latency[bucket] = median
    overall_median_latency[bucket] = np.mean(same_bucket_list)

    # mean_latency[bucket] = np.mean(same_bucket_list)
    nn = np.percentile([elem for elem in chunked_p95_times if elem != None], 99)
    nn_latency[bucket] = nn
    overall_nn_latency[bucket] = np.percentile(same_bucket_list, 99)

    # Calculate the standard error of the mean
    sem_p95 = stats.sem(chunked_p95_times)
    sem_med = stats.sem(chunked_med_times)

    # Calculate the confidence intervals
    ci_med = stats.t.interval(0.95, len(chunked_med_times)-1, loc=np.median(same_bucket_list), scale=sem_med)
    median_latency_ci[bucket] = ci_med
    cur_uci_med.append(ci_med[1]*1000)
    cur_lci_med.append(ci_med[0]*1000)
    cur_resp_med.append(median*1000)

    ci_nn = stats.t.interval(0.95, len(chunked_p95_times)-1, loc=np.percentile(same_bucket_list, 99), scale=sem_p95)
    nn_latency_ci[bucket] = ci_nn
    cur_uci_p95.append(ci_nn[1]*1000)
    cur_lci_p95.append(ci_nn[0]*1000)
    cur_resp_p95.append(nn*1000)

# Add error bars to the plot
ax1.errorbar(buckets, [overall_median_latency[bucket] * 1000 for bucket in buckets], 
             yerr=[((upper-lower) * 1000) for lower, upper in median_latency_ci.values()], 
             marker='.', capsize=5, color='blue', label="mean")

ax1.errorbar(buckets, [overall_nn_latency[bucket] * 1000 for bucket in buckets], 
             yerr=[((upper-lower) * 1000) for lower, upper in nn_latency_ci.values()],
             marker='.', capsize=5, color='orange', label="p99")

ax1.set_xlabel("Forecasting RPS")
ax1.set_ylabel("Forecasting Latency (ms)")

ax1.set_xticks(buckets)
ax1.set_xticklabels(buckets)


# create a second x-axis at the top of the plot
ax2 = ax1.twiny()

# calculate the new x-values for the top x-axis
new_xticks = [value*60 for value in buckets]
new_xticklabels = [value for value in buckets]

# set the xticks and xticklabels for the top x-axis
ax2.set_xticks(new_xticks)
ax2.set_xticklabels(new_xticks)
ax2.set_xlabel("Number of Applications")

# calculate the padding for the x-axis limits
padding = (max(buckets) - min(buckets)) * 0.05

# set the limits of both axes to be the same with padding
ax1.set_xlim([min(buckets) - padding, max(buckets) + padding])
ax2.set_xlim([min(new_xticks) - padding*60, max(new_xticks) + padding*60])
ax1.legend()

ax1.set_ylim([0, 30])

# add grid
ax1.grid(alpha=0.2)
ax2.grid(alpha=0.2)

plt.tight_layout()

fig.savefig("../output_plots/sec_7_hsp_inference_latency.pdf")