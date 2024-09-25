#include "../src/einsummable/memgraph.h"

#include "../src/einsummable/mgallocator.cc"
#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"

#include <fstream>

// void main01()
// {
//     std::cout << "trying to test new allocator_t" << std::endl;

//     /*Take an example that we have a device with 100 bytes memory*/
//     allocator_t allocator = allocator_t(100);

//     allocator.print();
//     DOUT("");

//     auto [o0, _0] = allocator.allocate(6);
//     auto [o1, _1] = allocator.allocate(4);
//     auto [o2, _2] = allocator.allocate(2);
//     auto [o3, _3] = allocator.allocate(7);
//     DOUT("_0" << _0);
//     DOUT("_1" << _1);
//     DOUT("_2" << _2);
//     DOUT("_3" << _3);
//     allocator.free(o0, 0);
//     allocator.free(o1, 0);
//     allocator.free(o2, 0);
//     allocator.free(o3, 0);
//     allocator.print();
//     auto [o5, _4] = allocator.allocate(10);
//     DOUT("_4 " << _4);
//     allocator.print();
// }

// void usage()
// {
//     DOUT("pi pj pk di dj dk np");
// }

// int main_(int argc, char** argv)
// {
//     if (argc != 8) {
//         usage();
//         return 1;
//     }

//     int      pi, pj, pk;
//     uint64_t di, dj, dk;
//     int      np;
//     try {
//         pi = parse_with_ss<int>(argv[1]);
//         pj = parse_with_ss<int>(argv[2]);
//         pk = parse_with_ss<int>(argv[3]);
//         di = parse_with_ss<uint64_t>(argv[4]);
//         dj = parse_with_ss<uint64_t>(argv[5]);
//         dk = parse_with_ss<uint64_t>(argv[6]);
//         np = parse_with_ss<int>(argv[7]);
//     } catch (...) {
//         std::cout << "Parse error." << std::endl << std::endl;
//         usage();
//         return 1;
//     }

//     auto g = three_dimensional_matrix_multiplication(pi, pj, pk, di, dj, dk, np);

//     auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());
//     {
//         std::cout << "tg.gv" << std::endl;
//         std::ofstream f("tg.gv");
//         taskgraph.print_graphviz(f);
//     }

//     // it could be the case that not all locs are actually used,
//     // for example 1 1 2 100 100 100 88
//     // Here only 2 locs will really be used, not all 88...
//     np = taskgraph.num_locs();

//     {
//         tuple<map<int, mem_t>, // input -> mem
//               map<int, mem_t>, // save -> mem
//               memgraph_t>
//             _info1 = memgraph_t::make_without_evict(
//                 taskgraph, {}, {allocator_strat_t::lowest_dependency, 1});
//         auto const& [_2, _3, memgraph] = _info1;

//         std::cout << "Printing to mm3d_mem_lowest_dep.gv" << std::endl;
//         std::ofstream f("mm3d_mem_lowest_dep.gv");
//         memgraph.print_graphviz(f);
//     }

//     {
//         tuple<map<int, mem_t>, // input -> mem
//               map<int, mem_t>, // save -> mem
//               memgraph_t>
//             _info1 = memgraph_t::make_without_evict(taskgraph, {}, {allocator_strat_t::first, 1});
//         auto const& [_2, _3, memgraph] = _info1;

//         std::cout << "Printing to mm3d_mem_first.gv" << std::endl;
//         std::ofstream f("mm3d_mem_first.gv");
//         memgraph.print_graphviz(f);
//     }

//     return 0;
// }

// int main02()
// {
//     auto settings = allocator_settings_t::default_settings();
//     settings.alignment_power = 4;

//     allocator_t allocator = allocator_t(100, settings);

//     auto [o0, _0] = allocator.allocate(1);
//     auto [o1, _1] = allocator.allocate(2);
//     auto [o2, _2] = allocator.allocate(3);
//     auto [o3, _3] = allocator.allocate(4);

//     DOUT(o0);
//     DOUT(o1);
//     DOUT(o2);
//     DOUT(o3);

//     allocator.print();
//     return 0;
// }

// int main03()
// {
//     auto settings = allocator_settings_t::default_settings();

//     allocator_t alo = allocator_t(100, settings);

//     auto [o0, _0] = alo.allocate(25);
//     auto [o1, _1] = alo.allocate(25);
//     auto [o2, _2] = alo.allocate(25);
//     auto [o3, _3] = alo.allocate(25);

//     DOUT("the allocator is full with four \"tensors\"");
//     alo.print();
//     DOUT("");

//     alo.free(o0, 0);
//     alo.free(o1, 1);
//     alo.free(o2, 2);
//     alo.free(o3, 3);

//     DOUT("the allocator is empty now");
//     alo.print();
//     DOUT("");

//     auto [o4, deps] = alo.allocate(100);
//     DOUT("the allocator is full again with one \"tensor\"");
//     alo.print();
//     DOUT("and that tensor depends on " << deps);

//     return 0;
// }

// int main04()
// {
//     auto        settings = allocator_settings_t::default_settings();
//     allocator_t alo = allocator_t(100, settings);

//     alo.allocate_at_without_deps(75, 25);
//     alo.allocate_at_without_deps(50, 25);
//     alo.allocate_at_without_deps(25, 25);
//     alo.allocate_at_without_deps(0, 25);

//     alo.print();

//     return 0;
// }


int free_from_middle_test()
{
    auto settings = allocator_settings_t::default_settings();

    allocator_t alo = allocator_t(100, settings);

    auto [o0, _0] = alo.allocate(25).value();
    auto [o1, _1] = alo.allocate(25).value();
    auto [o2, _2] = alo.allocate(25).value();
    auto [o3, _3] = alo.allocate(25).value();

    DOUT("the allocator is full with four \"tensors\"");
    alo.print();
    DOUT("");


    alo.free_from_middle(40,1);

    DOUT("done free from middle");
    alo.print();
    DOUT("");


    return 0;
}

int main(int argc, char** argv)
{
    // main_(argc, argv);
    // main03();
    free_from_middle_test();
}
