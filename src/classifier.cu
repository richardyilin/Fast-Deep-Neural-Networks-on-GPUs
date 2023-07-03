#include <iostream>
#include <cassert>
#include "dnn.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

using namespace std;

//Define the parameters if not defined externally
#ifndef Nn
  #define Nn 128  // Number of Output Layers
#endif
#ifndef Ni
  #define Ni 224  // Number of Input  Layers
#endif

# define NUM_THREADS_PER_BLOCK 512

VTYPE synapse[Nn][Ni] __attribute__((aligned(64)));
VTYPE neuron_i[Batch][Ni] __attribute__((aligned(64)));
VTYPE neuron_n[Batch][Nn] __attribute__((aligned(64)));
VTYPE neuron_n2[Batch][Nn] __attribute__((aligned(64)));

bool is_power_of_two(int n) {
  return (n > 0) && ((n & (n - 1)) == 0);
}

void fill_classifier(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Batch][Ni], 
    VTYPE (&neuron_n)[Batch][Nn],   VTYPE (&neuron_n2)[Batch][Nn]) {
    for(int n = 0; n < Nn; ++n) {
        for(int i = 0; i < Ni; ++i) {
          synapse[n][i] = static_cast <VTYPE> (rand()) / static_cast <VTYPE> (RAND_MAX) - 0.5f;
        }
    }
    for (int batch = 0; batch < Batch; batch++){
      for(int i = 0; i < Ni; ++i) {
        neuron_i[batch][i] = static_cast <VTYPE> (rand()) / static_cast <VTYPE> (RAND_MAX) - 0.5f;
      }
    }
    for (int batch = 0; batch < Batch; batch++){
      for(int n = 0; n < Nn; ++n) {
        neuron_n[batch][n] = 0;
        neuron_n2[batch][n] = 0;
      }
    }
}

void classifier_layer(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Batch][Ni], VTYPE (&neuron_n)[Batch][Nn]) {
  for (int batch = 0; batch < Batch; batch++){
    for (int n = 0; n < Nn; n++) {
      VTYPE temp=0;
      for (int i = 0; i < Ni; i++) {
        temp += synapse[n][i] * neuron_i[batch][i];
      }
      neuron_n[batch][n] = temp;
    }
  }
}

__global__ void classifier_GPU(VTYPE *d_synapse, VTYPE *d_neuron_i, VTYPE *d_neuron_n)
{
  // blockIdx.x is nn;
  // blockIdx.y is batch;
  int ni = threadIdx.x;

  __shared__ VTYPE cache[NUM_THREADS_PER_BLOCK];
  VTYPE neuron_n_value = 0;
  while (ni < Ni) {
    neuron_n_value += d_synapse[(blockIdx.x * Ni) + ni] * d_neuron_i[(blockIdx.y * Ni) + ni];
    ni += blockDim.x;
  }

  // Parallel prefix sum algorithm
  int num_active_thread = blockDim.x / 2;
  while (num_active_thread > 0) {
    if (threadIdx.x >= num_active_thread && threadIdx.x < num_active_thread * 2) {
      cache[threadIdx.x] = neuron_n_value;
    }
    __syncthreads();
    if (threadIdx.x < num_active_thread) {
      neuron_n_value += cache[threadIdx.x + num_active_thread];
    }
    num_active_thread /= 2;
  }

  //Thread 0 has the prefix sum, store it back to global memory
  if (threadIdx.x == 0) {
    d_neuron_n[(blockIdx.y * Nn) + blockIdx.x] = neuron_n_value;
  }
}

int main(void) {


    cout << "starting program\n";
    assert(is_power_of_two(NUM_THREADS_PER_BLOCK) && "NUM_THREADS_PER_BLOCK must be a power of 2.");
    cout << "initializing arrays on cpu\n\n";

    fill_classifier(synapse,neuron_i,neuron_n,neuron_n2);

    VTYPE *d_neuron_i, *d_synapse, *d_neuron_n2;

    cudaMalloc((void **)(&d_neuron_i), sizeof(VTYPE) * Batch * Ni);
    cudaMalloc((void **)(&d_synapse), sizeof(VTYPE) * Ni * Nn);
    cudaMalloc((void **)(&d_neuron_n2), sizeof(VTYPE) * Batch * Nn);

    cudaMemcpy(d_neuron_i, neuron_i, sizeof(VTYPE) * Batch * Ni, cudaMemcpyHostToDevice);
    cudaMemcpy(d_synapse, synapse, sizeof(VTYPE) * Ni * Nn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_neuron_n2, neuron_n2, sizeof(VTYPE) * Batch * Nn, cudaMemcpyHostToDevice);

    double cpu_time;
    cout << "CPU starts\n";

    begin_roi();
    classifier_layer(synapse,neuron_i,neuron_n);
    cpu_time = end_roi();

    cout << "CPU version completes!\n\n"; 

    cout << "GPU starts\n";

    double gpu_time;
    dim3 block_size(NUM_THREADS_PER_BLOCK);
    dim3 grid_size(Nn, Batch);
    begin_roi();
    classifier_GPU<<<grid_size, block_size>>>(d_synapse, d_neuron_i, d_neuron_n2);  
    cudaDeviceSynchronize();
    gpu_time = end_roi();


    cudaMemcpy(neuron_n2, d_neuron_n2, sizeof(VTYPE)*Batch * Nn, cudaMemcpyDeviceToHost);
    cuda_check_error();


    unsigned long long uNn = Nn;
    unsigned long long uNi = Ni;

    unsigned long long flop = uNn * uNi * Batch * 2; // float mul and add
    double gflops = flop / gpu_time / pow(10, 9);
    cout << "GPU achieves " << gflops << " GFLPOS\n";
    cout << "GPU is " << (cpu_time / gpu_time) << " faster than CPU simple version\n";
    cout << "GPU completes!\n\n";

    compare(*neuron_n,*neuron_n2,Batch * Nn);

    cout << "done\n\n";
    cudaFree(d_neuron_i);
    cudaFree(d_synapse);
    cudaFree(d_neuron_n2);
}