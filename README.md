# Fast-Deep-Neural-Networks-on-GPUs
## Table of contents

<!--ts-->
   * [Table of contents](#table-of-contents)
   * [Introduction](#introduction)
   * [Getting Started](#getting-started)
   * [Parameters](#parameters)
   * [Reference](#reference)
<!--te-->

## Introduction
   1. This project implements and evaluates 2D convolution layers and classifier layers in CUDA.
   2. For the pseudocode for a classifier layer and a convolutional layer, you can refer to Figure 5 and Figure 7 of the paper [DianNao: A Small-Footprint High-Throughput Accelerator for Ubiquitous Machine-Learning](https://dl.acm.org/doi/abs/10.1145/2654822.2541967?casa_token=aOU3QdqPD0kAAAAA:rQRR0yL-heT5taqG_RKvgCZtL4MGPDvJOApxIbpRM-h5K-IeNJFakNDZxDADVRNSY9EhnqAJhiw).

## Getting Started

   1. We first download the project and go to the [src](./src) folder.  

   ```sh
   git clone https://github.com/richardyilin/Fast-Deep-Neural-Networks-on-GPUs.git
   cd Fast-Deep-Neural-Networks-on-GPUs/src
   ```
   2. Then we need to make the makefile. There are 4 options to make:  

        1. ``make class1``
        2. ``make class2``
        3. ``make conv1``
        4. ``make conv2``

   3. These options correspond to different configurations of parameters of classifier and convolution layers. The parameters will be explained in the next section.  
   4. We execute the files we just made.  

        1. ``./class1``
        2. ``./class2``
        3. ``./conv1``
        4. ``./conv2``

## Parameters
   1. The parameters of options in the makefile are as follows:

        * class1: Ni=25088 Nn=4096
        * class2: Ni=4096 Nn=1024
        * conv1:  Nx=224 Ny=224 Kx=3  Ky=3  Ni=64   Nn=64  (stride=1)
        * conv2:  Nx=14  Ny=14  Kx=3  Ky=3  Ni=512 Nn=512  (stride=1)

   2. Definitions of parameters:

        * Ni/Nn -- Number of input feature maps.
        * Ni/Nn -- Number of output feature maps.
        * Nx -- Width of input feature maps.
        * Ny -- Height of input feature maps.
        * Kx -- Width of filters.
        * Ky -- Height of filters.
        * Batch -- Batch size.


## Reference
   1. Chen, Tianshi, et al. "Diannao: A small-footprint high-throughput accelerator for ubiquitous machine-learning." ACM SIGARCH Computer Architecture News 42.1 (2014): 269-284. 

