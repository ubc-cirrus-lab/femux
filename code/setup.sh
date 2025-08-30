#!/bin/bash
cd ../
mkdir data 
cd data
mkdir azure
cd azure
mkdir preproc_data azure_data forecaster_data transformed_data hypothesis_tests clustering
cd transformed_data 
mkdir concurrency app_conc_events && cd ..
cd forecaster_data
mkdir concurrency && cd ..
cd hypothesis_tests
mkdir concurrency && cd ..
cd clustering
mkdir concurrency && cd ..
cd azure_data
wget https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz
tar -xvf azurefunctions-dataset2019.tar.xz
