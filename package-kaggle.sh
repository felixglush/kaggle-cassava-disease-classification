#!/bin/bash

# this script packages my kaggle project files and models into 2 separate zip files for upload as "datasets" to Kaggle
exp_name=$1
zip -r models.zip trained-models/$exp_name
zip -r scripts.zip environment.yml *.py *.ipynb

mkdir ../cassava_models
mkdir ../cassava_scripts

mv models.zip ../cassava_models
mv scripts.zip ../cassava_scripts

kaggle datasets init -p ../cassava_models
kaggle datasets init -p ../cassava_scripts

echo "Fill in the datapackage.json file in the dataset folders"
echo "run: kaggle datasets create -p /path/to/dataset"
