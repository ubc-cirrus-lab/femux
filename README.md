# Instructions on Generating Simulation-based Results
## Setup Notes
- We configure our scripts to run on 48 cores and 140GB of Memory (on Ubuntu) to minimize runtime, but are parameterizable to any number of cores--16 core and 80GB works as well.
- We comment out Holt, Exponential Smoothing, and SETAR to significantly cut down forecasting simulation runtime which is the bottleneck for reproduction. Results do not change significantly as these forecasters are selected for under 5 percent of blocks.


## Preprocessing
We start with the azure 2019 dataset and clean, format, and partition the data for our simulations. 

1. Make some of the basic directories and download the azure 2019 dataset: `cd code & ./setup.sh` 
2. Preprocess the data and generate separate dataframes for
execution times, invocation counts, and application memory: each of the files in `/code/preprocess/`
3. Convert invocations per minute into average concurrency: `/code/transform/transform_azure.py`
4. Generate training and testing split for applications: `/code/gen_train_test_split.py`

## Simulation: Offline Forecasting and Features Used Later
1. Forecasting Simulations: `/code/forecasting/forecast.py`
2. Extracting Features: `/code/extract/feature_extraction.py`
3. Simulate the cold start and memory usage/allocation based on the simulated forecasts: `/code/results/gen_results.py`
4. Generate MAE values based on simualted forecasts (for comparing RUM and MAE): `/code/results/maes.py`

## Clustering Forecasters and Simulating Switching Performance
`clustering_pipeline.py` combines the clustering, and simulated performance of the FeMux prototype 
configured with the named features and forecasters that have already been extracted/simulated above.

# Plotting Results
1. Plotters are in `/plots/plots/plotters`, and all data in `/plots/data` can be unzipped using `/plots/setup.sh`.
2. To use generated simulation results instead of those we provide, can replace the files in `/plots/data` with those in `/data`
