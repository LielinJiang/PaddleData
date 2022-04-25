/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "image_decoder.h"

template <typename T>
__global__ void copy(const T* in,
                           T* out,
                           const int numel) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < numel;
       index += blockDim.x * gridDim.x) {
    out[index] = in[index];
  }

}

template <typename T>
void copy_kernelLauncher(const float* in,
                               T* out,
                               const int numel,
                               cudaStream_t stream){
  dim3 grid(64);
  dim3 block(64);
  copy<<<grid, block, 0, stream>>>(in, out, numel);
}
