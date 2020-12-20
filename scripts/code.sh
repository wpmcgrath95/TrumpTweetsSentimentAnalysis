#!/usr/bin/env bash
sleep 30

mkdir -p ./plots/
mkdir -p ./models/
mkdir -p ./data/

python3 ./python/LDA_Grouping.py
python3 ./python/Text_Blob.py
python3 ./python/Labeling_Model.py
python3 ./python/Cluster_Model.py