cmake_minimum_required(VERSION 3.2)

project(einsummable)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


add_library(einsummable
  STATIC
  src/einsummable/copyregion.cc
  src/einsummable/coster.cc
  src/einsummable/einsummable.cc
  src/einsummable/graph.cc
  src/einsummable/indexer.cc
  src/einsummable/partdim.cc
  src/einsummable/partition.cc
  src/einsummable/reference.cc
  src/einsummable/scalarop.cc
  src/einsummable/setup.cc
  src/einsummable/taskgraph.cc
)

add_executable(make_3dmatmul exps/make_3dmatmul.cc)
target_link_libraries(make_3dmatmul einsummable)

add_executable(reference exps/reference.cc)
target_link_libraries(reference einsummable)

add_executable(copyregion exps/copyregion.cc)
target_link_libraries(copyregion einsummable)

add_executable(coster exps/coster.cc)
target_link_libraries(coster einsummable)

add_executable(scalarop exps/scalarop.cc)
target_link_libraries(scalarop einsummable)

add_library(matrixgraph
  STATIC
  src/matrixgraph/matrixgraph.cc
  src/einsummable/einsummable.cc
  src/einsummable/graph.cc
  src/einsummable/partdim.cc
  src/einsummable/partition.cc
  src/einsummable/setup.cc
  src/einsummable/indexer.cc
  # TODO: einsummable/*.cc here fixes linking error in matgraphexp; why?
)

add_executable(matrixgraphexp exps/matrixgraph.cc)
target_link_libraries(matrixgraphexp einsummable)
target_link_libraries(matrixgraphexp matrixgraph)

if(${CPU_EXECUTION_ENGINE})
  find_package(MPI REQUIRED)

  include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

  add_library(cpu_execution_engine
    STATIC
    src/execution/cpu/execute.cc
    src/execution/cpu/mpi_class.cc
    src/execution/cpu/kernels.cc
    src/execution/cpu/permute.cc
  )

  add_executable(cpuexec exps/cpuexec.cc)
  target_link_libraries(cpuexec einsummable)
  target_link_libraries(cpuexec cpu_execution_engine)
  target_link_libraries(cpuexec ${MPI_CXX_LIBRARIES})

  add_executable(cpuexec_ew exps/cpuexec_ew.cc)
  target_link_libraries(cpuexec_ew einsummable)
  target_link_libraries(cpuexec_ew cpu_execution_engine)
  target_link_libraries(cpuexec_ew ${MPI_CXX_LIBRARIES})

  add_executable(cpuexec_touch exps/cpuexec_touch.cc)
  target_link_libraries(cpuexec_touch einsummable)
  target_link_libraries(cpuexec_touch cpu_execution_engine)
  target_link_libraries(cpuexec_touch ${MPI_CXX_LIBRARIES})

  add_executable(cpuexec_permute exps/cpuexec_permute.cc)
  target_link_libraries(cpuexec_permute einsummable)
  target_link_libraries(cpuexec_permute cpu_execution_engine)
  target_link_libraries(cpuexec_permute ${MPI_CXX_LIBRARIES})
endif()
