# Parallel-KMeans-HPC

Developed during the **2024–2025 academic year**, this project implements parallel versions of the K-Means clustering algorithm using Pthreads, OpenMP, and CUDA. It analyzes performance gains, scalability, and speedup across multi-core CPUs and GPUs, depending on thread/block configurations and dataset sizes.

The project includes:
- Sequential version – `kmeans_seq.c`
- Multi-threaded version using Pthreads – `kmeans_pthread.c`
- GPU-accelerated version using CUDA – `kmeans_cuda.cu`