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

find_package(Protobuf REQUIRED)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src/proto)
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

add_base_application("make_3dmatmul"   "exps" "make_3dmatmul")
add_base_application("reference"       "exps" "reference")
add_base_application("copyregion"      "exps" "copyregion")
add_base_application("scalarop"        "exps" "scalarop")
add_base_application("forwardsim"      "exps" "forwardsim")
add_base_application("allocator_test"  "exps" "allocator_test")

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

  add_cpu_application("cpuexec"    "exps" "cpuexec")
  #add_cpu_application("cpuexec_ff" "exps" "cpuexec_ff")
  add_cpu_application("cpuexec_ew" "exps" "cpuexec_ew")
  add_cpu_application("cpuexec_touch" "exps" "cpuexec_touch")
  add_cpu_application("cpuexec_permute" "exps" "cpuexec_permute")
  add_cpu_application("cpuexec_matmul" "exps" "cpuexec_matmul")
endif()

if(${GPU_EXECUTION_ENGINE})
  enable_language(CUDA)

  find_package(CUDA REQUIRED)
  find_package(CUDAToolkit)
  include_directories("${CUDA_INCLUDE_DIRS}")

  add_library(gpu_execution_engine
    STATIC
    ${BA_FILES}
    ${ES_FILES}
    ${MG_FILES}
    ${AP_FILES}
  )

  function(add_gpu_application exec dir filename)
  add_executable(${exec} ${dir}/${filename}.cc)
  target_link_libraries(${exec}
    gpu_execution_engine
    proto
    ${Protobuf_LIBRARIES}
    ${CUDA_LIBRARIES}
    -lcutensor)
  endfunction()

  # call add_gpu_application here
endif()
