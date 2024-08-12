#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../src/base/args.h"
#include "../../src/server/gpu/server.h"

#include "../../llama/gpu_llama.h"

#include <string>
#include <cstring>

namespace py = pybind11;

communicator_t gpu_mg_server_t::null_comm;

template <typename T>
void print_graphviz(T const& obj, string filename)
{
  std::ofstream f(filename);
  obj.print_graphviz(f);
  DOUT("printed " << filename);
}



graph_t matmul_graph(int pi, int pj, int pk, int di, int dj, int dk, int num_processors, int mem_size)
{
  auto g = three_dimensional_matrix_multiplication(
  pi,pj,pk, di,dj,dk, num_processors);
  return g.graph;
}

graph_constructor_t matmul_graph_con(int pi, int pj, int pk, int di, int dj, int dk, int num_processors, int mem_size)
{
  auto g = three_dimensional_matrix_multiplication(
  pi,pj,pk, di,dj,dk, num_processors);
  return g;
}

float scalar_to_float(scalar_t s) {
    return s.f32();
}

void print_dbuf(dbuffer_t dbuf) {
    DOUT(dbuf);
}   

taskgraph_t matmul_taskgraph(int pi, int pj, int pk, int di, int dj, int dk, int num_processors, int mem_size) {
  auto g = three_dimensional_matrix_multiplication(
  pi,pj,pk, di,dj,dk, num_processors);

  // print_graphviz(g.graph, "g.gv");

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());

  // print_graphviz(taskgraph, "tg.gv");

//   auto [_2, _3, memgraph] = memgraph_t::make(
//     taskgraph,
//     {},
//     vector<uint64_t>(num_processors, mem_size));

//   print_graphviz(memgraph, "mg.gv");

  return taskgraph;
}

dbuffer_t dbuf_from_numpy(uintptr_t ptr, uint64_t size, dtype_t dtype) {
    buffer_t buf = make_buffer_copy(make_buffer_reference((uint8_t *) (void *) ptr, size * dtype_size(dtype)));
    return dbuffer_t(dtype, buf);
}

PYBIND11_MODULE(PyEinsummable, m) {
    // Binding for gpu_mg_server_t
    py::class_<gpu_mg_server_t>(m, "server")
        .def(py::init<std::vector<uint64_t>, uint64_t>())
        .def("execute_memgraph", &gpu_mg_server_t::execute_memgraph)
        .def("execute_graph", &gpu_mg_server_t::execute_graph)
        .def("insert_tensor", py::overload_cast<int, const std::vector<uint64_t> &, dbuffer_t>(&gpu_mg_server_t::insert_tensor))
        .def("get_tensor", &gpu_mg_server_t::get_tensor_from_gid)
        .def("get_relation", &gpu_mg_server_t::get_relation)
        .def("get_gids", &gpu_mg_server_t::get_gids)
        .def("get_max_tid", &gpu_mg_server_t::get_max_tid)
        .def("insert_gid_without_data", &gpu_mg_server_t::insert_gid_without_data)
        .def("shutdown", &gpu_mg_server_t::shutdown);

    py::class_<relation_t>(m, "relation")
        .def("singleton", &relation_t::make_singleton);
    py::class_<placement_t>(m, "placement");
    
    // Binding for dbuffer_t
    py::class_<dbuffer_t>(m, "tensor", py::buffer_protocol())
        .def_buffer([](dbuffer_t &m) -> py::buffer_info {
              return py::buffer_info(
                  m.raw(),
                  dtype_size(m.dtype),
                  (std::string) "",
                  1,
                  {m.size() / dtype_size(m.dtype)},
                  {dtype_size(m.dtype)}
              );
        })
        .def("get", &dbuffer_t::get)
        .def("sum", &dbuffer_t::sum)
        .def("sumf64", &dbuffer_t::sum_to_f64);
        // .def(py::init([](py::buffer) {
        //     py::buffer_info info = b.request();
            
        // }));

    // Binding for graphs
    py::class_<memgraph_t>(m, "memgraph");
    py::class_<graph_t>(m, "eingraph")
        .def_readwrite("nodes", &graph_t::nodes)
        .def("get_inputs", &graph_t::get_inputs)
        .def("print", &graph_t::print);
    py::class_<taskgraph_t>(m, "taskgraph");
    py::class_<graph_t::node_t>(m, "node")
        .def_readwrite("inns", &graph_t::node_t::inns)
        .def_readwrite("outs", &graph_t::node_t::outs)
        .def_readwrite("op", &graph_t::node_t::op);
    py::class_<graph_constructor_t>(m, "graph_constructor")
        .def_readwrite("graph", &graph_constructor_t::graph)
        .def_readwrite("placements", &graph_constructor_t::placements);
    // py::class_<graph_t::op_t>(m, "op")
    //     .def("is_input", &graph_t::op_t::is_input);

    py::class_<scalar_t>(m, "scalar")
        .def("str", &scalar_t::str);
    py::enum_<dtype_t>(m, "dtype")
        .value("f16", dtype_t::f16)
        .value("f32", dtype_t::f32)
        .value("f64", dtype_t::f64)
        .value("c64", dtype_t::c64)
        .export_values();

    py::enum_<llama_size_t>(m, "llama_size")
        .value("B7", llama_size_t::B7)
        .value("B13", llama_size_t::B13)
        .value("B30", llama_size_t::B30)
        .value("B65", llama_size_t::B65);
    
    py::class_<gpu_llama_t>(m, "llama")
        .def(py::init<uint64_t, uint64_t, llama_size_t>())
        .def("train", &gpu_llama_t::train)
        .def("load_tensors", &gpu_llama_t::load_tensors);


    m.def("matmul_memgraph", &matmul_taskgraph, "Give the various graphs for a matrix multiplication of the given parameters");
    m.def("matmul_graph", &matmul_graph);
    m.def("matmul_graph_con", &matmul_graph_con);
    m.def("dbuf_from_numpy", &dbuf_from_numpy);
    m.def("scalar_to_float", &scalar_to_float);
    m.def("print_dbuf", &print_dbuf);
}
