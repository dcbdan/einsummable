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
#set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -pg -fno-omit-frame-pointer -DNDEBUG")#-fno-inline-functions -fno-inline-functions-called-once -fno-optimize-sibling-calls")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

# TODO: make sure protocol buffers are built in the correct order
#       with respect to other files. Sometimes make -j is required
#       when make fails
# Note: running make proto will build proto

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
add_base_application("memgraph_test"    "exps" "memgraph_test")
add_base_application("gwise"            "exps" "gwise")
add_base_application("mm"               "exps" "mm")
add_base_application("engine"           "exps" "engine")

function(add_llama_application exec filename)
add_executable(${exec} llama/${filename}.cc
  llama/modules.cc llama/misc.cc llama/builder.cc)
target_link_libraries(${exec} einsummable proto ${Protobuf_LIBRARIES})
endfunction()

add_llama_application("llama_main"  "main")
add_llama_application("llama_exp01" "exp01")

if(${CPU_EXECUTION_ENGINE})
  if(NOT MKL_THREADING)
    set(MKL_THREADING sequential)
  endif()

  add_compile_definitions(CPU_EXEC)

  # Tell MKL to use the best instructions; if
  # this instruction set it goes to the next best
  # and so on...
  # (MKL is bad at finding the right instruction set and
  #  this helps empirically)
  set(MKL_ENABLE_INSTRUCTIONS AVX512_E4)

  find_package(MKL REQUIRED)

  if(NOT DEFINED UCX_PATH)
    message( SEND_ERROR "Pleased define UCX_PATH")
  endif()
  set(UCX_DIR "${UCX_PATH}/lib/cmake/ucx")
  include_directories(${UCX_PATH}/include)
  find_package(UCX REQUIRED COMPONENTS ucp uct ucs)

  include_directories(${MKL_INCLUDE})

  file(GLOB EG_FILES ${PROJECT_SOURCE_DIR}/src/engine/*.cc)
  file(GLOB SE_FILES ${PROJECT_SOURCE_DIR}/src/server/*.cc)
  file(GLOB CP_FILES ${PROJECT_SOURCE_DIR}/src/engine/cpu/*.cc)

  # TODO: how do you make it so that not all the ES_FILES and MG_FILES
  #       have to be rebuilt?
  add_library(cpu_execution_engine
    STATIC
    ${BA_FILES}
    ${ES_FILES}
    ${MG_FILES}
    ${AP_FILES}
    ${EG_FILES}
    ${SE_FILES}
    ${CP_FILES}
  )

  function(add_cpu_application exec dir filename)
  add_executable(${exec} ${dir}/${filename}.cc)
  target_link_libraries(${exec}
    cpu_execution_engine
    proto
    ${Protobuf_LIBRARIES}
    MKL::MKL
    ucx::ucp ucx::ucs)
  endfunction()

  #add_cpu_application("cpuexec"         "exps" "cpuexec")
  #add_cpu_application("cpuexec_3d"      "exps" "cpuexec_3d")
  #add_cpu_application("cpuexec_ff"      "exps" "cpuexec_ff")
  #add_cpu_application("cpuexec_mm"      "exps" "cpuexec_mm")
  #add_cpu_application("cpuexec_touch"   "exps" "cpuexec_touch")
  #add_cpu_application("cpuexec_permute" "exps" "cpuexec_permute")
  #add_cpu_application("cpuexec_kernels" "exps" "cpuexec_kernels")
  #add_cpu_application("cpuexec_gen"     "exps" "cpuexec_gen")
  #add_cpu_application("cpumanager"      "exps" "cpumanager")
  #add_cpu_application("cpukerneltime"   "exps" "cpukerneltime")
  #add_cpu_application("cpumms"          "exps" "cpumms")
  add_cpu_application("cpu_engine"      "exps" "cpu_engine")
  add_cpu_application("cpu_engine_ml"   "exps" "cpu_engine_ml")

  function(add_cpu_llama_application exec filename)
    add_executable(${exec} llama/${filename}.cc
      llama/modules.cc llama/builder.cc llama/misc.cc llama/reader.cc)
    target_link_libraries(${exec}
      cpu_execution_engine
      proto
      ${Protobuf_LIBRARIES}
      MKL::MKL)
  endfunction()

  #add_cpu_llama_application("llama_cpu" "cpu")

endif()

if(${GPU_EXECUTION_ENGINE})
  enable_language(CUDA)

  find_package(CUDA 11.0 REQUIRED)
  include_directories("${CUDA_INCLUDE_DIRS}")
  include_directories(${CUTENSOR_ROOT}/include)
  include_directories(${PROJECT_SOURCE_DIR}/src/execution/gpu)

  # set the right architecture
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")
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
    cublasLt
    cublas)

  endfunction()

  # call add_gpu_application here
  ## add_gpu_application("cutensorexp"  "exps" "cutensorexp")
  add_gpu_application("test_alignment"  "exps" "test_alignment")

  add_gpu_application("basicTest" "exps" "GPU_test")
  # add_gpu_application("ffTest" "exps" "GPU_ff_test")
endif()
