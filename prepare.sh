#!/usr/bin/env bash

mkdir -p data
cd data
echo "Downloading data at `pwd`"
base_url="https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC"
wget "$base_url/Ising2DFM_reSample_L40_T=All.pkl"
wget "$base_url/Ising2DFM_reSample_L40_T=All_labels.pkl"
cd ..

echo "Done"
