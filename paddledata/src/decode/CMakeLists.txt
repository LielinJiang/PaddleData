# cmake_minimum_required(VERSION 3.16)
# project(paddledata  LANGUAGES C CXX CUDA)

# find_package(CUDA 10.1 REQUIRED)
# set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

# list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

# list(APPEND COMMON_HEADER_DIRS
#   ${PROJECT_SOURCE_DIR}
#   ${CUDA_PATH}/include)

# set(COMMON_LIB_DIRS
#   ${CUDA_PATH}/lib64
# )

# INCLUDE(ExternalProject)

set(THIRD_PARTY_NAME opencv)
set(THIRD_PARTY_PATH  "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
    "A path setting third party libraries download & build directories.")
set(THIRD_PARTY_CACHE_PATH     "${CMAKE_SOURCE_DIR}"    CACHE STRING
    "A path cache third party source code to avoid repeated download.")

ExternalProject_Add(
  extern_opencv
  GIT_REPOSITORY    https://github.com/opencv/opencv.git
  GIT_TAG           4.5.4
  PREFIX            ${THIRD_PATH}
  SOURCE_DIR        ${THIRD_PATH}/source/${THIRD_PARTY_NAME}
  UPDATE_COMMAND    ${FT_UPDATE_COMMAND}  # PATCH_COMMAND     ${FT_PATCH_COMMAND}
  BINARY_DIR        ${THIRD_PATH}/build/${THIRD_PARTY_NAME}
  INSTALL_COMMAND   ""
  CMAKE_ARGS        -DCMAKE_BUILD_TYPE=RELEASE
                    -DVIBRANTE_PDK:STRING=/   
                    -DBUILD_LIST=core,improc,imgcodecs 
                    -DBUILD_ENCODER=${WITH_ENCODER} 
                    -DBUILD_SHARED_LIBS=ON 
                    -DON_INFER=${ON_INFER} 
                    -DWITH_CUDA=OFF 
                    -DWITH_1394=OFF 
                    -DWITH_IPP=OFF 
                    -DWITH_OPENCL=OFF 
                    -DWITH_GTK=OFF 
                    -DBUILD_JPEG=OFF 
                    -DWITH_JPEG=ON 
                    -DBUILD_TIFF=OFF 
                    -DWITH_TIFF=ON 
                    -DBUILD_JASPER=OFF 
                    -DBUILD_DOCS=OFF       
                    -DBUILD_TESTS=OFF       
                    -DBUILD_PERF_TESTS=OFF       
                    -DBUILD_PNG=ON       
                    -DWITH_WEBP=ON       
                    -DBUILD_opencv_cudalegacy=OFF       
                    -DBUILD_opencv_stitching=OFF       
                    -DWITH_TBB=OFF       
                    -DWITH_OPENMP=OFF       
                    -DWITH_PTHREADS_PF=OFF       
                    -DBUILD_EXAMPLES=OFF       
                    -DBUILD_opencv_java=OFF       
                    -DBUILD_opencv_python2=OFF       
                    -DBUILD_opencv_python3=OFF       
                    -DWITH_PROTOBUF=OFF       
                    -DWITH_FFMPEG=OFF       
                    -DWITH_GSTREAMER=OFF       
                    -DWITH_GSTREAMER_0_10=OFF       
                    -DWITH_VTK=OFF       
                    -DWITH_OPENEXR=OFF       
                    -DINSTALL_C_EXAMPLES=OFF       
                    -DINSTALL_TESTS=OFF       
                    -DVIBRANTE=TRUE       
                    -DWITH_CSTRIPES=OFF   
                    -DWITH_QUIRC=OFF 
                    -DWITH_ADE=OFF   
                    # -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}       
                    # VERBOSE=1
)

# if (${CUDA_VERSION} GREATER_EQUAL 11.0)
#   message(STATUS "Add DCUDA11_MODE")
#   add_definitions("-DCUDA11_MODE")
# endif()
include_directories(
  ${COMMON_HEADER_DIRS}
)

# list(APPEND COMMON_LIB_DIRS /usr/local/mpi/lib)

link_directories(
  ${COMMON_LIB_DIRS}
)

list(APPEND decoding_op_files random_roi_generator.cc image_decoder.cc image_decoder.cu decode.cc decode.cu)

add_definitions(-DNDEBUG)
add_definitions(-DPADDLE_CUDA)
# Default is 1 in standard c++ when using gcc8.2
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)


if(NOT PY_CMD)
  set(PYTHON_PATH "python" CACHE STRING "Python path")
else()
  set(PYTHON_PATH ${PY_CMD} CACHE STRING "Python path")
endif()

execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import paddle; print(paddle.sysconfig.get_include())"
                RESULT_VARIABLE _INC_PYTHON_SUCCESS
                OUTPUT_VARIABLE _INC_PYTHON_VALUES)
if (NOT _INC_PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Python config Error.")
endif()
string(REGEX REPLACE ";" "\\\\;" _INC_PYTHON_VALUES ${_INC_PYTHON_VALUES})
string(REGEX REPLACE "\n" ";" _INC_PYTHON_VALUES ${_INC_PYTHON_VALUES})
list(GET _INC_PYTHON_VALUES 0 PY_INCLUDE_DIR)

list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})
list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR}/third_party)

include_directories(
  ${COMMON_HEADER_DIRS}
)

execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import paddle; print(paddle.sysconfig.get_lib())"
                RESULT_VARIABLE _LIB_PYTHON_SUCCESS
                OUTPUT_VARIABLE _LIB_PYTHON_VALUES)
if (NOT _LIB_PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Python config Error.")
endif()
string(REGEX REPLACE ";" "\\\\;" _LIB_PYTHON_VALUES ${_LIB_PYTHON_VALUES})
string(REGEX REPLACE "\n" ";" _LIB_PYTHON_VALUES ${_LIB_PYTHON_VALUES})
list(GET _LIB_PYTHON_VALUES 0 PY_LIB_DIR)
list(APPEND COMMON_LIB_DIRS ${PY_LIB_DIR})

link_directories(
  ${COMMON_LIB_DIRS}
)

include_directories(${PY_INCLUDE_DIR})
include_directories(${PY_INCLUDE_DIR}/third_party)
message("PYTHON LIB DIR:" ${PY_LIB_DIR})
if(EXISTS ${PY_LIB_DIR}/libpaddle_custom_op.so)
  set(lib_link
    -lpaddle_custom_op
  )
endif()

if(EXISTS ${PY_LIB_DIR}/../fluid/core_avx.so)
  set(lib_link
    -l:core_avx.so
  )
  link_directories(
    ${PY_LIB_DIR}/../fluid/
  )
else()
  set(lib_link
    -l:core_noavx.so
  )
  link_directories(
    ${PY_LIB_DIR}/../fluid/
  )
endif()

set(decode_lib_link
  -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lnvjpeg
)


add_definitions(-DPADDLE_WITH_CUDA)
add_definitions(-DEIGEN_USE_GPU)
add_definitions(-DPADDLE_USE_DSO)


add_library(image_decode_op SHARED ${decoding_op_files})
add_dependencies(image_decode_op extern_opencv)
target_link_libraries(image_decode_op PRIVATE ${lib_link} ${decode_lib_link})
