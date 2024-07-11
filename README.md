# SMaT: (S)parse (Ma)trix Matrix (T)ensor Core-accelerated library

## Requirements

### Hardware 
We run our experiments on the Swiss National Computing Center’s Ault compute cluster. Each node
is equipped with a single NVIDIA A100-SXM4-40GB GPU,
and AMD EPYC 7742 @ 2.25GHz CPU. The A100 driver
version is 530.30.02.

### Software 
All experiments were executed using the GCC
12.3.0 compiler, NVIDIA nvcc v12.0, NVIDIA cuSPARSE
v12.0, NVIDIA CUDA Toolkit v12.0, Python 3.9, and the
following Python libraries: Pandas, Matplotlib, Numpy, Scipy,
and Seaborn


To create a conda environment:
```bash
conda env create -f smat_env.yml
conda activate smat
```

## Datasets
For preparing the matrices run the following:

- SuiteSparse Collection:
```bash
python download_suitesparse.py
```
- Synthetic band matrices:
```bash
python generate_matrices.py
```


## Compiling
In order to compile the library:
```bash
cd src/cuda_hgemm
source compile.sh
```

## Running The Code
Further details and scripts for reproducing experiments can be found: [here](./src/scripts/README.md).