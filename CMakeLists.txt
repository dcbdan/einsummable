cmake_minimum_required(VERSION 3.2)

project(einsummable)

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

#add_library(matgraph
#  STATIC
#  src/matgraph/matgraph.cc
#  src/einsummable/einsummable.cc
#  src/einsummable/graph.cc
#  src/einsummable/partdim.cc
#  src/einsummable/partition.cc
#  src/einsummable/setup.cc
#  src/einsummable/indexer.cc
#  # TODO: einsummable/*.cc here fixes linking error in matgraphexp; why?
#)
#
#add_executable(matgraphexp exps/matgraph.cc)
#target_link_libraries(matgraphexp einsummable)
#target_link_libraries(matgraphexp matgraph)
#
#if(${CPU_EXECUTION_ENGINE})
#  find_package(MPI REQUIRED)
#
#  include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
#
#  add_library(cpu_execution_engine
#    STATIC
#    src/execution/cpu/execute.cc
#    src/execution/cpu/mpi_class.cc
#  )
#
#  add_executable(cpuexec exps/cpuexec.cc)
#  target_link_libraries(cpuexec einsummable)
#  target_link_libraries(cpuexec cpu_execution_engine)
#  target_link_libraries(cpuexec ${MPI_CXX_LIBRARIES})
#endif()
