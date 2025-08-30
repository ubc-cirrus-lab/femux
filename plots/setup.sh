# check if venv directory exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Virtual environment exists. Activating..."
    source venv/bin/activate
fi

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