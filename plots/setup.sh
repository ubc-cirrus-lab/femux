cp -r ../data/azure/results/* ./data/
cp -r ../data/azure/train_test_split ./data/
cp ../data/azure/preproc_data/app_total_inv_exec_12_days.pickle ./data/

# unzip files
# go inside each subdirectory of ./data and unzip the files inside
for dir in ./data/*/
do
    dir=${dir%*/}
    echo "Unzipping files in $dir"
    for file in $dir/*.zip
    do
        unzip -o $file -d $dir
    done
done