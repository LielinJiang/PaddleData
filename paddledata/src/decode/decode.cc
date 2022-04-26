#include "paddle/extension.h"

#include <vector>
#include <iostream>
#include <nvjpeg.h>
#include <opencv2/opencv.hpp>
#include "random_roi_generator.h"
#include "image_decoder.h"
using namespace cv;

#define CHECK_CPU_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> 
decode_cuda_forward(const std::vector<paddle::Tensor>& x,
                    int num_threads,
                    int local_rank,
                    int64_t program_id);

#endif

static cudaStream_t nvjpeg_stream = nullptr;
static nvjpegHandle_t nvjpeg_handle = nullptr;

void InitNvjpegImage(nvjpegImage_t* img) {
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
    img->channel[c] = nullptr;
    img->pitch[c] = 0;
  }
};

std::vector<paddle::Tensor> DecodeForward(
                        const std::vector<paddle::Tensor>& x,
                        const int num_threads,
                        const int local_rank,
                        const int64_t program_id,
                        const int64_t host_memory_padding,
                        const int64_t device_memory_padding,
                        float aspect_ratio_min,
                        float aspect_ratio_max,
                        float area_min,
                        float area_max
                        ) {

  auto* decode_pool = 
    ImageDecoderThreadPoolManager::Instance()->GetDecoderThreadPool(
                        program_id, num_threads, local_rank,
                        static_cast<size_t>(host_memory_padding),
                        static_cast<size_t>(device_memory_padding));
  int batch_size = x.size();

  AspectRatioRange aspect_ratio_range{aspect_ratio_min, aspect_ratio_max};
  AreaRange area_range{area_min, area_max};

  auto* generators = GeneratorManager::Instance()->GetGenerators(
                        program_id, batch_size, aspect_ratio_range,
                        area_range);
                  
  if (nvjpeg_handle == nullptr) {
      nvjpegStatus_t create_status = nvjpegCreateSimple(&nvjpeg_handle);

      if(create_status != NVJPEG_STATUS_SUCCESS)
        PD_THROW("nvjpegCreateSimple failed: ", create_status);
  }

  nvjpegJpegState_t nvjpeg_state;
  nvjpegStatus_t state_status = nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state);

  int components;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];

  std::vector<paddle::Tensor> outs;
  outs.resize(x.size());

  for (int i = 0; i < x.size(); i++) {
    auto* x_data = x[i].data<uint8_t>();
    size_t x_numel = static_cast<size_t>(x[i].size());
    ImageDecodeTask task = {
      .bit_stream = x_data,
      .bit_len = x_numel,
      .outs = &outs,
      .index = i,
      .roi_generator = generators->at(i).get(),
      .place = paddle::PlaceType::kGPU
    };
    decode_pool->AddTask(std::make_shared<ImageDecodeTask>(task));
  }

  decode_pool->RunAll(true);
  
  std::vector<paddle::Tensor> outs_tansposed;
  return outs;
}

std::vector<std::vector<int64_t>> DecodeForwardInferShape(
  const std::vector<std::vector<int64_t>>& input_shapes
) {
  return input_shapes;
}

PD_BUILD_OP(custom_decode)
    .Inputs({paddle::Vec("X")})
    .Outputs({paddle::Vec("Out")})
    .Attrs({"num_threads: int",
             "local_rank: int",
             "program_id: int64_t",
             "host_memory_padding: int64_t",
             "device_memory_padding: int64_t",
             "aspect_ratio_min: float",
             "aspect_ratio_max: float",
             "area_min: float",
             "area_max: float"})
    .SetKernelFn(PD_KERNEL(DecodeForward))
    .SetInferShapeFn(PD_INFER_SHAPE(DecodeForwardInferShape));
