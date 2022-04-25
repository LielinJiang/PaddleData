#include "paddle/extension.h"

#define CHECK_CUDA_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")


std::vector<paddle::Tensor> 
decode_cuda_forward(const std::vector<paddle::Tensor>& x,
                    int num_threads,
                    std::string mode,
                    int local_rank,
                    int64_t program_id) {
  CHECK_CUDA_INPUT(x[0]);

  std::vector<paddle::Tensor> outs;
  for (int i = 0; i < x.size(); i++) {
    outs.push_back(x[i]);
  }

  return outs;
}
