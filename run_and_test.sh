#!/bin/bash

cd ./src/cuda_hgemm

./build.sh

cd ../..

./src/cuda_hgemm/output/bin/hgemm 