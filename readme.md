# Spatha Matrix Multiplication Acceleration Project

## Introduction
This project focuses on the transformation of the matrix multiplication  $X \times W = Y$ where $X$ is a **dense matrix** and  $W$ is a **sparse matrix**. The key transformation applied is converting the multiplication to $W^T \times X^T = Y^T$, and then utilizing the capabilities of the **Spatha** library to accelerate this computation.

![pic1](For_the_example_notebook.png "example")

### Why Use Spatha Library?
Spatha library offers support for The **V:N:M** (VENOM) format, which allows for the execution of arbitrary N:M ratios on Sparse Tensor Cores (SPTCs). Typically, SPTCs natively support only 2:4 patterns (indicating 50% sparsity). By employing the VENOM format, Spatha enhances the flexibility and efficiency of sparse matrix operations on supported hardware.

### Why the Transposition?
The Sparse Tensor Cores used in Spatha are optimized for operations where a sparse matrix multiplies a dense matrix. To align with this hardware optimization when a dense matrix needs to multiply a sparse matrix, a transposition approach is utilized. By transposing both $X$ and $W$, the multiplication $W^T \times X^T$ fits the hardware's operational design. The resultant matrix $Y^T$ is then transposed back to obtain the desired result $Y$.

## Installation and Execution