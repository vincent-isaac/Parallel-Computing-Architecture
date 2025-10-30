# Parallel Computing Architecture

This repository contains a series of GPU programming and parallel computing experiments implemented using CUDA-C. These experiments demonstrate fundamental GPU concepts including unified memory, shared memory, thread hierarchy, warp divergence, reduction operations, and performance benchmarking.


## 📁 Experiments Overview

### EXP 01: Matrix Addition Using Unified Memory
- **Description**: Implements matrix addition using CUDA unified memory to simplify memory management between CPU and GPU.

### EXP 02: Warp Divergence and Sum Reduction
- **Description**: Demonstrates how warp divergence affects GPU performance and implements parallel sum reduction.

### EXP 03: Matrix Summation with 2D Grid and 2D Blocks
- **Description**: Performs matrix addition using a 2D grid of blocks and 2D thread blocks for structured data parallelism.

### EXP 04: Matrix Transposition Using Shared Memory
- **Description**: Implements matrix transpose with shared memory to reduce global memory access overhead and improve performance.

### EXP 05: GPU-Based Vector Summation
- **Description**: Compares CPU and GPU vector addition and evaluates performance differences.

### EXP 06: Matrix Multiplication Using CUDA
- **Description**: Implements GPU-based matrix multiplication and measures execution time compared to CPU computation.

### EXP 07: Image Grayscale Conversion (CUDA)
- **Description**: A CUDA-based project performing either grayscale image conversion or face detection depending on implementation.

## 🛠️ Technologies Used

- **Language**: CUDA C/C++, C/C++
- **Frameworks / APIs**: NVIDIA CUDA Toolkit
- **Tools**: NVCC Compiler, Visual Studio / NVIDIA Nsight 