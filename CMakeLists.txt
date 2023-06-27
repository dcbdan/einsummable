cmake_minimum_required(VERSION 3.2)

project(einsummable)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set (CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-sign-compare")
#set(CMAKE_CXX_FLAGS "-Wmaybe-uninitialized")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# TODO: make sure protocol buffers are built in the correct order
#       with respect to other files. Sometimes make -j is required
#       when make fails

find_package(Protobuf REQUIRED)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src/proto)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/half-2.2.0/include)
include_directories(${Protobuf_INCLUDE_DIRS})

add_subdirectory(src/proto)

file(GLOB BA_FILES ${PROJECT_SOURCE_DIR}/src/base/*.cc)
file(GLOB ES_FILES ${PROJECT_SOURCE_DIR}/src/einsummable/*.cc)
file(GLOB MG_FILES ${PROJECT_SOURCE_DIR}/src/matrixgraph/*.cc)
file(GLOB AP_FILES ${PROJECT_SOURCE_DIR}/src/autoplace/*.cc)

add_library(einsummable
  STATIC
  ${BA_FILES}
  ${ES_FILES}
  ${MG_FILES}
  ${AP_FILES}
)

function(add_base_application exec dir filename)
add_executable(${exec} ${dir}/${filename}.cc)
target_link_libraries(${exec} einsummable proto ${Protobuf_LIBRARIES})
endfunction()

add_base_application("make_3dmatmul"    "exps" "make_3dmatmul")
add_base_application("reference"        "exps" "reference")
add_base_application("copyregion"       "exps" "copyregion")
add_base_application("scalarop"         "exps" "scalarop")
add_base_application("forwardsim"       "exps" "forwardsim")
add_base_application("allocator_test"   "exps" "allocator_test")
add_base_application("gwriter"          "exps" "gwriter")
add_base_application("deepff"           "exps" "deepff")
add_base_application("base_test"        "exps" "base_test")
add_base_application("einsummable_test" "exps" "einsummable_test")

function(add_llama_application exec filename)
add_executable(${exec} llama/${filename}.cc llama/modules.cc llama/misc.cc)
target_link_libraries(${exec} einsummable proto ${Protobuf_LIBRARIES})
endfunction()

add_llama_application("llama_main" "main")

if(${CPU_EXECUTION_ENGINE})
  find_package(MPI REQUIRED)

  find_package(MKL REQUIRED)

  include_directories(${MKL_INCLUDE})

  include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

  file(GLOB CP_FILES ${PROJECT_SOURCE_DIR}/src/execution/cpu/*.cc)

  # TODO: how do you make it so that not all the ES_FILES and MG_FILES
  #       have to be rebuilt?
  add_library(cpu_execution_engine
    STATIC
    ${BA_FILES}
    ${ES_FILES}
    ${MG_FILES}
    ${AP_FILES}
    ${CP_FILES}
  )

  function(add_cpu_application exec dir filename)
  add_executable(${exec} ${dir}/${filename}.cc)
  target_link_libraries(${exec}
    cpu_execution_engine
    proto
    ${Protobuf_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    MKL::MKL)
  endfunction()

  add_cpu_application("cpuexec"         "exps" "cpuexec")
  add_cpu_application("cpuexec_3d"      "exps" "cpuexec_3d")
  add_cpu_application("cpuexec_ff"      "exps" "cpuexec_ff")
  add_cpu_application("cpuexec_mm"      "exps" "cpuexec_mm")
  add_cpu_application("cpuexec_touch"   "exps" "cpuexec_touch")
  add_cpu_application("cpuexec_permute" "exps" "cpuexec_permute")
  add_cpu_application("cpuexec_kernels" "exps" "cpuexec_kernels")
  add_cpu_application("cpuexec_gen"     "exps" "cpuexec_gen")

  function(add_cpu_llama_application exec filename)
  add_executable(${exec} llama/${filename}.cc llama/modules.cc llama/builder.cc llama/misc.cc)
  target_link_libraries(${exec}
    cpu_execution_engine
    proto
    ${Protobuf_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    MKL::MKL)
  endfunction()

  add_cpu_llama_application("llama_cpu" "cpu")

endif()

if(${GPU_EXECUTION_ENGINE})
  enable_language(CUDA)

  find_package(CUDA 11.0 REQUIRED)
  include_directories("${CUDA_INCLUDE_DIRS}")
  include_directories(${CUTENSOR_ROOT}/include)

  # set the right architecture
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

  file(GLOB CP_FILES ${PROJECT_SOURCE_DIR}/src/execution/gpu/*.cc)
  file(GLOB CU_FILES ${PROJECT_SOURCE_DIR}/src/execution/gpu/*.cu)

  add_library(cutensor SHARED IMPORTED)
  set_target_properties(cutensor PROPERTIES
    IMPORTED_LOCATION "${CUTENSOR_ROOT}/lib/11.0/libcutensor.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_ROOT}/include")

  add_library(gpu_execution_engine
    STATIC
    ${BA_FILES}
    ${ES_FILES}
    ${MG_FILES}
    ${AP_FILES}
    ${CP_FILES}
    ${CU_FILES}
  )

  function(add_gpu_application exec dir filename)
  add_executable(${exec} ${dir}/${filename}.cc)

  target_link_libraries(${exec}
    gpu_execution_engine
    proto
    ${Protobuf_LIBRARIES}
    ${CUDA_LIBRARIES}
    cutensor
    cudart
    cublasLt)

  endfunction()

  # call add_gpu_application here
  add_gpu_application("test_alignment"  "exps" "test_alignment")
  add_gpu_application("gpuexec_contraction"  "exps" "gpuexec_contraction" )
endif()
