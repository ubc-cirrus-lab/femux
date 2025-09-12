# Copy files and directories
cp -r ../data/azure/results/* ./data/
cp -r ../data/azure/train_test_split ./data/
cp ../data/azure/preproc_data/app_total_inv_exec_12_days.pickle ./data/

# Define a list of directories to ignore
ignore_dirs=("train_test_split" "mae" "hashapps_by_size")

# Unzip files
for dir in ./data/*/
do
    dir_name=$(basename "$dir")
    
    # Check if the directory name is in the ignore list
    case " ${ignore_dirs[@]} " in
        *" $dir_name "*)
            # Do nothing
            ;;
        *)
            # Unzip files if the directory is not ignored
            for file in "$dir"/*.zip
            do
                if [ -f "$file" ]; then
                    unzip -o "$file" -d "$dir"
                fi
            done
            ;;
    esac
done        
