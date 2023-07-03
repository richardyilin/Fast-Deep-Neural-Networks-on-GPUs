#pragma once

#include <random>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <functional>
#include <iomanip>

#include <inttypes.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define VTYPE float

inline void cuda_check_error() {
  auto err = cudaGetLastError();
  if(err) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(0);
  }
}

void fill_random(VTYPE *array, size_t size) {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_real_distribution<> dis(-0.5, 0.5);
  for(size_t i = 0; i < size; ++i) {
    array[i] = dis(gen);
  }
}

__attribute__ ((noinline)) double timeit(std::function<void ()> f) {
  auto start = std::chrono::high_resolution_clock::now();
  f();
  std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;
  std::cout << std::left << std::setw(12) << diff.count() << " sec(s) elapsed." << std::endl;
  std::cout << std::left << std::setw(12) << (diff.count() / Batch) << " sec(s) elapsed per batch." << std::endl;
  return diff.count();
}

// template <typename F>
// __attribute__ ((noinline)) void CUDA_timeit(F f) {
//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
//   cudaEventRecord(start);
//   f();
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);

//   float exec_time;
//   cudaEventElapsedTime(&exec_time, start, stop);
//   std::cout << std::left << std::setw(12) << exec_time / 1000.0 << " sec(s) elapsed." << std::endl;
//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);
// }

void compare(VTYPE* neuron1, VTYPE* neuron2, int size) {
  bool error = false;
  for(int i = 0; i < size; ++i) {
      VTYPE diff = neuron1[i] - neuron2[i];
      if(diff>0.001f || diff <-0.001f) {
      error = true; 
      break;
    }
  }
  if(error) {
    for(int i = 0; i < size; ++i) {
      std::cout << i << " " << neuron1[i] << ":" << neuron2[i];;

      VTYPE diff = neuron1[i] - neuron2[i];
      if(diff>0.001f || diff <-0.001f) {
        std::cout << " \t\tERROR";
      }
      std::cout << "\n";
    }
  } else {
    std::cout << "results match\n";
  }
}

static __inline__ uint64_t gettime(void) { 
  struct timeval tv; 
  gettimeofday(&tv, NULL); 
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec)); 
} 

static uint64_t usec;

__attribute__ ((noinline))  void begin_roi() {
  usec=gettime();
}

__attribute__ ((noinline))  double end_roi()   {
  double time;
  time = (gettime()-usec) /1000000.0;
  std::cout << "elapsed (sec): " << time << "\n";
  std::cout << "elapsed per batch (sec): " << time / Batch << "\n";
  return time;
}