#include <iostream>
#include <string>
#include <functional>
#include "dnn.cuh"
#include <cmath>

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#ifndef Tnn
  //Tiling Sizes
  #define Tnn 32
  #define Tn  16
  #define Ti  16
  
  #define Ty  8
  #define Tx  8
#endif

#define NYPAD (Ny+Ky-1)
#define NXPAD (Nx+Kx-1)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (Ky * Kx * Nn * Ni)

#define NUM_THREADS_PER_BLOCK 512
#define NUM_OUTPUTS_PER_BLOCK_X_DIM (Tx-Kx+1) //  Tx-Kx+1 is the number of output a block can compute in x dimension. Add Tx-Kx to Nx function as ceil
#define NUM_OUTPUTS_PER_BLOCK_Y_DIM (Ty-Ky+1)
#define NUM_BLOCKS_X_DIM ((Nx + Tx-Kx)  / (Tx-Kx+1)) //  Tx-Kx+1 is the number of output a block can compute in x dimension. Add Tx-Kx to Nx function as ceil
#define NUM_BLOCKS_Y_DIM_PER_BATCH ((Ny + Ty-Ky)  / (Ty-Ky+1))

#define NUM_SYNAPSE_PER_BLOCK (Ky * Kx * Ni * Tn)
#define NUM_NEURON_I_PER_BLOCK (Ty * Tx * Ni)
#define NUM_NEURON_N_PER_BLOCK (NUM_OUTPUTS_PER_BLOCK_Y_DIM * NUM_OUTPUTS_PER_BLOCK_X_DIM * Tn)
#define NUM_OP_PER_OUTPUT_NEURON (Ky * Kx * Ni)
#if (NUM_THREADS_PER_BLOCK / NUM_NEURON_N_PER_BLOCK) < NUM_OP_PER_OUTPUT_NEURON
  #define NUM_THREAD_PER_NEURON_N (NUM_THREADS_PER_BLOCK / NUM_NEURON_N_PER_BLOCK)
#else
  #define NUM_THREAD_PER_NEURON_N NUM_OP_PER_OUTPUT_NEURON
#endif

VTYPE synapse[Ky][Kx][Nn][Ni] __attribute__((aligned(64))); // synapse (filter)
VTYPE neuron_i[Batch][NYPAD][NXPAD][Ni] __attribute__((aligned(64))); // input neurons
VTYPE neuron_n[Batch][NYSCL][NXSCL][Nn] __attribute__((aligned(64))); // output neurons
VTYPE neuron_n1[Batch][NYSCL][NXSCL][Nn] __attribute__((aligned(64))); // output neurons
VTYPE neuron_n2[Batch][NYSCL][NXSCL][Nn] __attribute__((aligned(64))); // output neurons

void fill_convolution_shared_simple(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                                    VTYPE (&neuron_i)[Batch][NYPAD][NXPAD][Ni],
                                    VTYPE (&neuron_n)[Batch][NYSCL][NXSCL][Nn],
                                    VTYPE (&neuron_n1)[Batch][NYSCL][NXSCL][Nn],
                                    VTYPE (&neuron_n2)[Batch][NYSCL][NXSCL][Nn]) {
  for(int yy = 0; yy < Ky; ++yy) {
    for(int xx = 0; xx < Kx; ++xx) {
      for(int nn = 0; nn < Nn; ++nn) {
        for(int ni = 0; ni < Ni; ++ni) {
          synapse[yy][xx][nn][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        } } } }

  for (int batch = 0; batch < Batch; batch++){
    for(int yy = 0; yy < NYPAD; ++yy) {
      for(int xx = 0; xx < NXPAD; ++xx) {
        for(int ni = 0; ni < Ni; ++ni) {
          neuron_i[batch][yy][xx][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }  }  } }

  for (int batch = 0; batch < Batch; batch++){
    for(int yy = 0; yy < NYSCL; ++yy) {
      for(int xx = 0; xx < NXSCL; ++xx) {
        for(int nn = 0; nn < Nn; ++nn) {
          neuron_n[batch][yy][xx][nn] = 0;
    }  }  } }

  for (int batch = 0; batch < Batch; batch++){
    for(int yy = 0; yy < NYSCL; ++yy) {
      for(int xx = 0; xx < NXSCL; ++xx) {
        for(int nn = 0; nn < Nn; ++nn) {
          neuron_n1[batch][yy][xx][nn] = 0;
    }  }  } }

  for (int batch = 0; batch < Batch; batch++){
    for(int yy = 0; yy < NYSCL; ++yy) {
      for(int xx = 0; xx < NXSCL; ++xx) {
        for(int nn = 0; nn < Nn; ++nn) {
          neuron_n2[batch][yy][xx][nn] = 0;
    }  }  } }
}


void convolution_tiled(VTYPE (&synapse)[Ky][Kx][Nn][Ni],
                       VTYPE (&neuron_i)[Batch][NYPAD][NXPAD][Ni],
                       VTYPE (&neuron_n)[Batch][NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn] = {0};

  for (int batch = 0; batch < Batch; batch++){
    for (int yy = 0; yy < Ny; yy += Ty) {
      for (int xx = 0; xx < Nx; xx += Tx) {
        for (int nnn = 0; nnn < Nn; nnn += Tnn) {
          int yout = yy / Sy;
          for (int y = yy; y < min(yy + Ty, Ny); y += Sy) { // tiling for y;
            int xout = xx / Sx;

            for (int x = xx; x < min(xx + Tx, Nx); x += Sx) { // tiling for x;

              for (int nn = nnn; nn < min(nnn + Tnn, Nn); nn += Tn) {
                memset(sum + nn, 0, Tn * sizeof(VTYPE));

                for (int ky = 0; ky < Ky; ky++) {  // sliding window;
                  for (int kx = 0; kx < Kx; kx++) {
                    VTYPE sum_sc;

                    for (int ii = 0; ii < Ni; ii += Ti) {
                      for (int n = nn; n < min(nn + Tn, Nn); n++) {
                        sum_sc=0;
                        for (int i = ii; i < min(ii + Ti, Ni); i++) {
                          VTYPE sv = synapse[ky][kx][n][i];
                          VTYPE nv = neuron_i[batch][ky + y][kx + x][i];
                          sum_sc += sv * nv;
                        }
                        sum[n] += sum_sc;
                      }
                    }
                  }
                }
                for (int n = nn; n < min(nn + Tn, Nn); n++) {
                  neuron_n[batch][yout][xout][n] = sum[n];
                }
              }
              xout++;
            }
            yout++;
          }
        }
      }
    }
  }
}

void convolution(VTYPE (&synapse)[Ky][Kx][Nn][Ni],
                       VTYPE (&neuron_i)[Batch][NYPAD][NXPAD][Ni],
                       VTYPE (&neuron_n)[Batch][NYSCL][NXSCL][Nn]) {
  VTYPE sum, sv, nv;
  for (int batch = 0; batch < Batch; batch++){
    for (int y = 0; y < NYSCL; y++) {
      for (int x = 0; x < NXSCL; x++) {
        for (int nn = 0; nn < Nn; nn++){
          sum = 0.0;
          for (int ky = 0; ky < Ky; ky++){
            for (int kx = 0; kx < Kx; kx++){
              for (int ni = 0; ni < Ni; ni++){
                sv = synapse[ky][kx][nn][ni];
                nv = neuron_i[batch][ky + y][kx + x][ni];
                sum += sv * nv;
              }
            }
          }
          neuron_n[batch][y][x][nn] = sum;
        }
      }
    }
  }
}

__device__ void load_synapse(VTYPE *synapse, VTYPE (&synapse_tiled)[Ky][Kx][Tn][Ni], int& nn_begin_neuron_n_global)
{
  int x, y, ni, nn, nn_neuron_n_global;
  for (int synapse_idx = threadIdx.x; synapse_idx < NUM_SYNAPSE_PER_BLOCK; synapse_idx += blockDim.x) {
    ni = synapse_idx % Ni;
    nn = (synapse_idx / Ni) % Tn;
    nn_neuron_n_global = nn_begin_neuron_n_global + nn;
    x = (synapse_idx / Ni / Tn) % Kx;
    y = (synapse_idx / Ni / Tn / Kx) % Ky;
    if (nn_neuron_n_global < Nn) {
      synapse_tiled[y][x][nn][ni] = synapse[((((y * Kx + x) * Nn) + nn_neuron_n_global) * Ni + ni)];
    }
  }
}

__device__ void load_input_neuron(VTYPE *neuron_i, VTYPE (&neuron_i_tiled)[Ty][Tx][Ni], int& x_begin_neuron_i_global, int& y_begin_neuron_i_global, int& batch)
{
  int x, y, ni, x_neuron_i_local, y_neuron_i_local;
  for (int neuron_i_idx = threadIdx.x; neuron_i_idx < NUM_NEURON_I_PER_BLOCK; neuron_i_idx += blockDim.x) {
    x_neuron_i_local = (neuron_i_idx / Ni) % Tx;
    y_neuron_i_local = (neuron_i_idx / Ni / Tx) % Ty;
    x = x_begin_neuron_i_global + x_neuron_i_local;
    y = y_begin_neuron_i_global + y_neuron_i_local;
    if (x < NXPAD && y < NYPAD) {
      ni = neuron_i_idx % Ni;
      neuron_i_tiled[y_neuron_i_local][x_neuron_i_local][ni] = neuron_i[(((batch * NYPAD + y) * NXPAD + x) * Ni + ni)];
    }
  }
}

#if (NUM_THREAD_PER_NEURON_N < 2)

__device__ void get_output_neuron_single_thread_multiple_output_neurons(VTYPE (&synapse_tiled)[Ky][Kx][Tn][Ni], VTYPE (&neuron_i_tiled)[Ty][Tx][Ni],\
VTYPE *neuron_n, int& x_begin_neuron_i_global, int& y_begin_neuron_i_global, int& nn_begin_neuron_n_global, int& batch)
{
  int x, y, nn, nn_neuron_n_global, x_neuron_n_conv, x_neuron_n_global, y_neuron_n_conv, y_neuron_n_global;
  VTYPE synapse_temp, neuron_i_temp, neuron_n_value;

  for (int neuron_n_idx = threadIdx.x; neuron_n_idx < NUM_NEURON_N_PER_BLOCK; neuron_n_idx += blockDim.x) {
    nn = neuron_n_idx % Tn;
    x_neuron_n_conv = (neuron_n_idx / Tn) % NUM_OUTPUTS_PER_BLOCK_X_DIM;
    y_neuron_n_conv = (neuron_n_idx / Tn) / NUM_OUTPUTS_PER_BLOCK_X_DIM;
    x_neuron_n_global = x_begin_neuron_i_global + x_neuron_n_conv; // x_begin_neuron_i_global = x_begin_neuron_n_global since input and output has the same starting point
    y_neuron_n_global = y_begin_neuron_i_global + y_neuron_n_conv;
    nn_neuron_n_global = nn_begin_neuron_n_global + nn;
    if (x_neuron_n_global < NXSCL && y_neuron_n_global < NYSCL && nn_neuron_n_global < Nn) {
      // sliding window;
      neuron_n_value = 0;
      for (int ky = 0; ky < Ky; ky++) {
        for (int kx = 0; kx < Kx; kx++) {
          for (int i = 0; i < Ni; i++) {
            x = x_neuron_n_conv + kx;
            y = y_neuron_n_conv + ky;
            synapse_temp = synapse_tiled[ky][kx][nn][i];
            neuron_i_temp = neuron_i_tiled[y][x][i];
            neuron_n_value += synapse_temp * neuron_i_temp;
          }
        }
      }
      neuron_n[(((batch * NYSCL + y_neuron_n_global) * NXSCL + x_neuron_n_global) * Nn + nn_neuron_n_global)] = neuron_n_value;
    }
  }
}

#else

__device__ void get_output_neuron_single_output_neuron_multiple_threads(VTYPE (&synapse_tiled)[Ky][Kx][Tn][Ni], VTYPE (&neuron_i_tiled)[Ty][Tx][Ni],\
VTYPE *neuron_n, int& x_begin_neuron_i_global, int& y_begin_neuron_i_global, int& nn_begin_neuron_n_global, int& batch) {

  int neuron_n_idx = threadIdx.x / NUM_THREAD_PER_NEURON_N;
  int thread_idx_in_neuron_n = threadIdx.x % NUM_THREAD_PER_NEURON_N;
  int nn = neuron_n_idx % Tn;
  int x_neuron_n_conv = (neuron_n_idx / Tn) % NUM_OUTPUTS_PER_BLOCK_X_DIM;
  int y_neuron_n_conv = (neuron_n_idx / Tn) / NUM_OUTPUTS_PER_BLOCK_X_DIM;
  int x_neuron_n_global = x_begin_neuron_i_global + x_neuron_n_conv;
  int y_neuron_n_global = y_begin_neuron_i_global + y_neuron_n_conv;
  int nn_neuron_n_global = nn_begin_neuron_n_global + nn;

  __shared__ VTYPE cache[NUM_NEURON_N_PER_BLOCK][NUM_THREAD_PER_NEURON_N];

  if (x_neuron_n_global < NXSCL && y_neuron_n_global < NYSCL && nn_neuron_n_global < Nn && neuron_n_idx < NUM_NEURON_N_PER_BLOCK) { // the output is in the valid region
    int x, y, kx, ky, i;
    VTYPE synapse_temp, neuron_i_temp, neuron_n_value;

    neuron_n_value = 0;
    for (int neuron_i_idx = thread_idx_in_neuron_n; neuron_i_idx < NUM_OP_PER_OUTPUT_NEURON; neuron_i_idx += NUM_THREAD_PER_NEURON_N) { // a thread computes partial sum for itself
      i = neuron_i_idx % Ni;
      kx = (neuron_i_idx / Ni) % Kx;
      ky = (neuron_i_idx / Ni) / Kx;
      x = x_neuron_n_conv + kx;
      y = y_neuron_n_conv + ky;
      synapse_temp = synapse_tiled[ky][kx][nn][i];
      neuron_i_temp = neuron_i_tiled[y][x][i];
      neuron_n_value += synapse_temp * neuron_i_temp;
    }
    int last_num_active_thread = NUM_THREAD_PER_NEURON_N;
    int num_active_thread = (NUM_THREAD_PER_NEURON_N + 1) / 2;
    int addend_thread_idx;
    while (last_num_active_thread > 1) {
      if (thread_idx_in_neuron_n >= num_active_thread && thread_idx_in_neuron_n < last_num_active_thread) {
        cache[neuron_n_idx][thread_idx_in_neuron_n] = neuron_n_value;
      }
      __syncthreads();
      addend_thread_idx = thread_idx_in_neuron_n + num_active_thread;
      if (thread_idx_in_neuron_n < num_active_thread && addend_thread_idx < last_num_active_thread) {
        neuron_n_value += cache[neuron_n_idx][addend_thread_idx];
      }
      last_num_active_thread = num_active_thread;
      num_active_thread = (num_active_thread + 1) / 2;
    }
    if (thread_idx_in_neuron_n == 0) {
      neuron_n[(((batch * NYSCL + y_neuron_n_global) * NXSCL + x_neuron_n_global) * Nn + nn_neuron_n_global)] = neuron_n_value;
    }
  }
}

#endif

__global__ void convolution_GPU(VTYPE *synapse,
                                VTYPE *neuron_i,
                                VTYPE *neuron_n) {

  __shared__ VTYPE synapse_tiled[Ky][Kx][Tn][Ni];
  __shared__ VTYPE neuron_i_tiled[Ty][Tx][Ni];

  int batch;
  int x_begin_neuron_i_global;
  int y_begin_neuron_i_global;
  int nn_begin_neuron_n_global;

  batch = blockIdx.y / NUM_BLOCKS_Y_DIM_PER_BATCH;
  x_begin_neuron_i_global = blockIdx.x * NUM_OUTPUTS_PER_BLOCK_X_DIM;
  y_begin_neuron_i_global = (blockIdx.y % NUM_BLOCKS_Y_DIM_PER_BATCH) * NUM_OUTPUTS_PER_BLOCK_Y_DIM;
  nn_begin_neuron_n_global = blockIdx.z * Tn;

  load_synapse(synapse, synapse_tiled, nn_begin_neuron_n_global);
  load_input_neuron(neuron_i, neuron_i_tiled, x_begin_neuron_i_global, y_begin_neuron_i_global, batch);

  __syncthreads();

  // compute convolution and write back

  #if (NUM_THREAD_PER_NEURON_N < 2)

  get_output_neuron_single_thread_multiple_output_neurons(synapse_tiled, neuron_i_tiled, neuron_n,\
  x_begin_neuron_i_global, y_begin_neuron_i_global, nn_begin_neuron_n_global, batch);

  #else

  get_output_neuron_single_output_neuron_multiple_threads(synapse_tiled, neuron_i_tiled, neuron_n,\
  x_begin_neuron_i_global, y_begin_neuron_i_global, nn_begin_neuron_n_global, batch);

  #endif
}

int main(void) {

  cout << "starting program\n";
  cout << "initializing arrays on cpu\n";

  fill_convolution_shared_simple(synapse, neuron_i, neuron_n, neuron_n1, neuron_n2);

  VTYPE *d_neuron_i, *d_synapse, *d_neuron_n2;

  cudaMalloc((void **)(&d_neuron_i), sizeof(VTYPE) * Batch * NXPAD * NYPAD * Ni);
  cudaMalloc((void **)(&d_synapse), sizeof(VTYPE) * Kx * Ky * Ni * Nn);
  cudaMalloc((void **)(&d_neuron_n2), sizeof(VTYPE) * Batch * NXSCL * NYSCL * Nn);

  cudaMemcpy(d_neuron_i, neuron_i, sizeof(VTYPE) * Batch * NXPAD * NYPAD * Ni, cudaMemcpyHostToDevice);
  cudaMemcpy(d_synapse, synapse, sizeof(VTYPE) * Kx * Ky * Ni * Nn, cudaMemcpyHostToDevice);
  cudaMemcpy(d_neuron_n2, neuron_n2, sizeof(VTYPE) * Batch * NXSCL * NYSCL * Nn, cudaMemcpyHostToDevice);

  double cpu_simple_time, cpu_tiled_time;
  cout << "\nCPU Simple version:\n";
  cpu_simple_time = timeit([&]() {
      convolution(synapse, neuron_i, neuron_n);
  });

  cout << "\nCPU Tiled version:\n";
  cpu_tiled_time = timeit([&]() {
      convolution_tiled(synapse, neuron_i, neuron_n1);
  });

  compare(***neuron_n, ***neuron_n1, Batch * NYSCL * NXSCL * Nn);

  double gpu_time;
  dim3 block_size(NUM_THREADS_PER_BLOCK);
  dim3 grid_size(NUM_BLOCKS_X_DIM, NUM_BLOCKS_Y_DIM_PER_BATCH * Batch, ((Nn + Tn - 1) / Tn));
  cout << "\nGPU starts!\n";
  begin_roi();
  convolution_GPU<<<grid_size, block_size>>>(d_synapse, d_neuron_i, d_neuron_n2);  
  cudaDeviceSynchronize();
  gpu_time = end_roi();

  cudaMemcpy(neuron_n2, d_neuron_n2, sizeof(VTYPE)* Batch *NXSCL*NYSCL*Nn, cudaMemcpyDeviceToHost);
  cuda_check_error();

  unsigned long long uNx = Nx;
  unsigned long long uNy = Ny;
  unsigned long long uNn = Nn;
  unsigned long long uNi = Ni;
  unsigned long long uKx = Kx;
  unsigned long long uKy = Ky;

  unsigned long long flop = (uNx * uNy * uNn) * (uKy * uKx * uNi) * Batch * 2; // float mul and add
  double gflops = flop / gpu_time / pow(10, 9);
  cout << "GPU achieves " << gflops << " GFLPOS\n";
  cout << "GPU is " << (cpu_simple_time / gpu_time) << " faster than CPU simple version\n";
  cout << "GPU is " << (cpu_tiled_time / gpu_time) << " faster than CPU tiled version\n";
  cout << "GPU completes!\n";

  compare(***neuron_n,***neuron_n2,Batch * NXSCL*NYSCL*Nn);

  cout << "done\n\n";

  cudaFree(d_neuron_i);
  cudaFree(d_synapse);
  cudaFree(d_neuron_n2);
}