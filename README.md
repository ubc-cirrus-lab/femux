Nima Nasiri, Nalin Munshi, Simon D Moser, Marius Pirvu, Vijay Sundaresan, Daryl Maier, Thatta Premnath, Norman Böwing, Sathish Gopalakrishnan, and Mohammad Shahrad, "In-Production Characterization of an Open Source Serverless Platform and New Scaling Strategies", 2026 ACM European Conference on Computer Systems (EuroSys '26).

# Instructions on Generating Simulation-based Results
Our artifact includes the code necessary to reproduce all simulated results for FeMux--eventually producing FeMux's results for Figures 7, 8, 10, and 11. 

## Setup Notes
- We recommend using Ubuntu 20.04 or newer with at least 150GB of disk space, and Python +3.10 with the dependencies defined in `requirements.txt`. Further, we recommend configuring our scripts to run on 48 cores and 140GB of Memory to minimize runtime, but they are parameterizable to any number of cores--we set them to 16 core and 80GB by default.
 - Time estimates below are given on 16 core machines, where core count can be updated globally by changing `num_workers = 16` or by changing the parameter at the bottom of each parallelizable script.
- We comment out Holt, Exponential Smoothing, and SETAR to significantly cut down forecasting simulation runtime which is the bottleneck for reproduction. Results do not change significantly as these forecasters are selected for under 5 percent of blocks. 
 - To save more time, we also configure FFT to run on application-level traces which affects some of the final results. For our FeMux prototype, we use function-level traces for FFT and the IceBreaker comparison. This can be seen in the initialization function of `/code/forecasting/forecasting_sim.py`.
- All scripts should be run from the directory they are defined in.

## Preprocessing
We start with the azure 2019 dataset and clean, format, and partition the data for our simulations.

1. Make some of the basic directories and download the azure 2019 dataset: `/code/setup.sh`
2. Preprocess the data and generate separate dataframes for
execution times, invocation counts, and application memory: each of the files in `/code/preprocess/`, starting with `/code/preprocess/preprocess_data.py`
3. Convert invocations per minute into average concurrency (<24h): `/code/transform/transform_azure.py`
4. Generate training and testing split for applications: `/code/gen_train_test_split.py`

## Simulation: Offline Forecasting and Features Used Later
1. Forecasting Simulations (~120h): `/code/forecasting/forecast.py`
2. Extracting Features (~4h): `/code/extract/feature_extraction.py`
3. Simulate the cold start and memory usage/allocation based on the simulated forecasts (<8h): `/code/results/gen_results.py`
4. Generate MAE values based on simulated forecasts (for comparing RUM and MAE): `/code/results/maes.py`

## Clustering Forecasters and Simulating Switching Performance
`clustering_pipeline.py` combines the clustering, and simulated performance of the FeMux prototype 
configured with the named features and forecasters that have already been extracted/simulated above (~1h).
 - Users with more than 128 cores need to set the OPENBLAS_NUM_THREADS environment variable to a number lesser than 128

# Plotting Results
1. `/plots/setup.sh` copies over generated results from `/data/azure` to `/plots/data`, and additionally unzips the pre-generated data we provide. 
2. Plotters are in `/plots/plots/plotters` and should be run within the directory. Each plotter has its corresponding plot output to a pdf file in `/plots/plots/output_plots`
 - Each plotter (e.g., `/plots/plots/plotters/sec_4_diff_size_apps.py`) corresponds to a figure in the respective section of the paper, with a short description of the result in the filename.
