# Parallel K-Means HPC

Developed during the 2024–2025 academic year, this compact C/CUDA study of the K-Means clustering algorithm compares a sequential baseline with parallel CPU and GPU implementations. The repository is designed for academic exploration of how parallelism affects runtime, scalability, and clustering quality on multidimensional point datasets.

## Overview

K-Means is an unsupervised machine learning algorithm that groups data points into k clusters by repeatedly:

1. Assigning each point to the nearest centroid.
2. Recomputing centroid positions as the mean of points in each cluster.
3. Repeating until centroids stabilize or a maximum iteration limit is reached.

This project includes three implementations:

- `kmeans_seq.c` — sequential version
- `kmeans_pthread.c` — multithreaded version using POSIX threads (Pthreads)
- `kmeans_cuda.c` — GPU-accelerated version using CUDA

The goal is to compare raw execution time and scaling behavior across serial, multi-core CPU, and GPU execution for the same clustering task.

## Project structure

```text
.
├── README.md
├── kmeans_seq.c
├── kmeans_pthread.c
├── kmeans_cuda.c
├── K-Means Clustering Parallel Implementations with Pthreads and CUDA.pdf
└── data files (user-provided; for example: data2.txt)
```

## Requirements

### Sequential and Pthreads builds

- GCC or Clang
- pthread support
- math library (`-lm`)

### CUDA build

- NVIDIA GPU
- CUDA Toolkit installed and configured
- `nvcc` available in `PATH`

## Building the project

### 1) Sequential baseline

```bash
gcc -O3 -std=c11 -lm kmeans_seq.c -o kmeans_seq
```

### 2) Pthreads version

```bash
gcc -O3 -std=c11 -pthread -lm kmeans_pthread.c -o kmeans_pthread
```

### 3) CUDA version

```bash
nvcc kmeans_cuda.c -o kmeans_cuda
```

## Running the programs

### Sequential version

```bash
./kmeans_seq
```

The program reads a dataset file, asks for the number of clusters `k`, and prompts for initial centroid coordinates.

### Pthreads version

```bash
./kmeans_pthread
```

This version prompts for:

- number of points `n`
- number of clusters `k`
- file name containing the dataset
- initial centroid coordinates

It then partitions the work across a fixed number of threads and measures execution time.

### CUDA version

```bash
./kmeans_cuda
```

This version requires:

- dataset file name
- number of clusters `k`
- initial centroid coordinates

It launches kernels to assign points to clusters and update centroids on the GPU.

## Input data format

Each point should be stored as a pair of floating-point coordinates in 2D space:

```text
0.50 2.10
1.20 3.40
2.10 1.80
...
```

One point per line, with `x y` values separated by whitespace.

## Example workflow

```bash
# Build
 gcc -O3 -std=c11 -lm kmeans_seq.c -o kmeans_seq
 gcc -O3 -std=c11 -pthread -lm kmeans_pthread.c -o kmeans_pthread
 nvcc kmeans_cuda.c -o kmeans_cuda

# Run each implementation on the same dataset
 ./kmeans_seq
 ./kmeans_pthread
 ./kmeans_cuda
```

## Notes

- The source files are intentionally simple and educational. They are not production-optimized libraries.
- The Pthreads and CUDA versions rely on manual input for cluster initialization and dataset loading.
- Performance can vary depending on hardware, dataset size, thread count, and GPU architecture.
- The CUDA implementation uses a fixed `threads_per_block` size and a straightforward reduction of partial sums on the device.

## Academic context

This repository was created to explore parallel computing concepts in a classic clustering problem. It is especially useful for comparing:

- sequential runtime
- CPU parallel speedup from Pthreads
- GPU acceleration using CUDA kernels and atomic reduction

## License

This project is provided for educational and research purposes. See the project files for usage context and academic references.

## Related material

The accompanying PDF in this repository documents the implementation and performance analysis for the parallel K-Means versions discussed here.
