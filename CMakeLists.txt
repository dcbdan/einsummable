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

add_library(einsummable
  STATIC
  src/einsummable/cluster.cc
  src/einsummable/copyregion.cc
  src/einsummable/einsummable.cc
  src/einsummable/forwardsim.cc
  src/einsummable/graph.cc
  src/einsummable/indexer.cc
  src/einsummable/loadbalanceplace.cc
  src/einsummable/memgraph.cc
  src/einsummable/partdim.cc
  src/einsummable/partition.cc
  src/einsummable/reference.cc
  src/einsummable/scalarop.cc
  src/einsummable/setup.cc
  src/einsummable/taskgraph.cc
  src/einsummable/timeplot.cc
  src/einsummable/twolayergraph.cc
  src/matrixgraph/matrixgraph.cc
  src/matrixgraph/ff.cc
)

function(add_base_application exec dir filename)
add_executable(${exec} ${dir}/${filename}.cc)
target_link_libraries(${exec} einsummable proto ${Protobuf_LIBRARIES})
endfunction()

add_base_application("make_3dmatmul" "exps" "make_3dmatmul")
add_base_application("reference"     "exps" "reference")
add_base_application("copyregion"    "exps" "copyregion")
add_base_application("scalarop"      "exps" "scalarop")
add_base_application("forwardsim"    "exps" "forwardsim")
add_base_application("autograph"     "exps" "autograph")

#if(${CPU_EXECUTION_ENGINE})
#  find_package(MPI REQUIRED)
#
#  find_package(MKL REQUIRED)
#
#  include_directories(${MKL_INCLUDE})
#
#  include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
#
#  add_library(cpu_execution_engine
#    STATIC
#    src/execution/cpu/execute.cc
#    src/execution/cpu/mpi_class.cc
#    src/execution/cpu/kernels.cc
#    src/execution/cpu/permute.cc
#    src/einsummable/copyregion.cc
#    src/einsummable/einsummable.cc
#    src/einsummable/graph.cc
#    src/einsummable/indexer.cc
#    src/einsummable/partdim.cc
#    src/einsummable/partition.cc
#    src/einsummable/reference.cc
#    src/einsummable/scalarop.cc
#    src/einsummable/setup.cc
#    src/einsummable/taskgraph.cc
#    src/matrixgraph/matrixgraph.cc
#    src/matrixgraph/ff.cc
#    src/einsummable/einsummable.cc
#    src/einsummable/graph.cc
#    src/einsummable/partdim.cc
#    src/einsummable/partition.cc
#    src/einsummable/setup.cc
#    src/einsummable/indexer.cc
#    # TODO: einsummable/*.cc here fixes linking error in matgraphexp; why?
#  )
#
#  add_executable(cpuexec exps/cpuexec.cc)
#  target_link_libraries(cpuexec einsummable)
#  target_link_libraries(cpuexec cpu_execution_engine)
#  target_link_libraries(cpuexec ${MPI_CXX_LIBRARIES})
#  target_link_libraries(cpuexec MKL::MKL)
#
#  add_executable(cpuexec_ff exps/cpuexec_ff.cc)
#  target_link_libraries(cpuexec_ff einsummable)
#  target_link_libraries(cpuexec_ff cpu_execution_engine)
#  target_link_libraries(cpuexec_ff ${MPI_CXX_LIBRARIES})
#  target_link_libraries(cpuexec_ff MKL::MKL)
#
#  add_executable(cpuexec_ew exps/cpuexec_ew.cc)
#  target_link_libraries(cpuexec_ew einsummable)
#  target_link_libraries(cpuexec_ew cpu_execution_engine)
#  target_link_libraries(cpuexec_ew ${MPI_CXX_LIBRARIES})
#  target_link_libraries(cpuexec_ew MKL::MKL)
#
#  add_executable(cpuexec_touch exps/cpuexec_touch.cc)
#  target_link_libraries(cpuexec_touch einsummable)
#  target_link_libraries(cpuexec_touch cpu_execution_engine)
#  target_link_libraries(cpuexec_touch ${MPI_CXX_LIBRARIES})
#  target_link_libraries(cpuexec_touch MKL::MKL)
#
#  add_executable(cpuexec_permute exps/cpuexec_permute.cc)
#  target_link_libraries(cpuexec_permute einsummable)
#  target_link_libraries(cpuexec_permute cpu_execution_engine)
#  target_link_libraries(cpuexec_permute ${MPI_CXX_LIBRARIES})
#  target_link_libraries(cpuexec_permute MKL::MKL)
#
#  add_executable(cpuexec_matmul exps/cpuexec_matmul.cc)
#  target_link_libraries(cpuexec_matmul einsummable)
#  target_link_libraries(cpuexec_matmul cpu_execution_engine)
#  target_link_libraries(cpuexec_matmul ${MPI_CXX_LIBRARIES})
#  target_link_libraries(cpuexec_matmul MKL::MKL)
#endif()
