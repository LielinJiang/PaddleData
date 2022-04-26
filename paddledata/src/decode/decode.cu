#include "paddle/extension.h"

std::vector<paddle::Tensor> 
decode_cuda_forward(const std::vector<paddle::Tensor>& x,
                    int num_threads,
                    int local_rank,
                    int64_t program_id) {

  std::vector<paddle::Tensor> outs;

  return outs;
}
