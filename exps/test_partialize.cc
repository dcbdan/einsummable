#include "../src/einsummable/reference.h"

#include "../src/einsummable/taskgraph.h"
#include <fstream>

int main(){
    taskgraph_t tg;
    dtype_t dtype = default_dtype();

    // case1: two different inputs for partialize
    int a1 = tg.insert_input(0, dtype, {50});
    int a2 = tg.insert_input(0, dtype, {50});
    // int b  = tg.insert_input(0, dtype, {100});

    int x = tg.new_partial(0, dtype, {100});
    tg.add_to_partial(x, a1,
      touch_t {
        .selection = { { 50, 100, 0, 0, 50 } },
        .castable = std::nullopt,
        .dtype = dtype
      },
      false);
    tg.add_to_partial(x, a2,
      touch_t {
        .selection = { { 50, 100, 0, 50, 50 } },
        .castable = std::nullopt,
        .dtype = dtype
      },
      false);

    // verify using reference_partialize
    dbuffer_t dataBufferA = make_dbuffer(dtype, 100);
    dataBufferA.iota(0);
    dbuffer_t dataBufferB = make_dbuffer(dtype, 100);
    dataBufferB.iota(100);
    map<int, dbuffer_t> inn_data{
      {0, dataBufferA},
      {1, dataBufferB}
    };

    // auto const& part = tg.nodes[1].op.get_partialize();
    // dbuffer_t out = reference_partialize(part,inn_data);
    // DOUT("reference_partialize: " << out);





    // // case2: one inputs that can not be simplified
    // int b1 = tg.insert_input(0, dtype, {100,100});
    // int b = tg.new_partial(0, dtype, {100,100});
    // // two halves of b both copy first half of b1
    // tg.add_to_partial(b, b1,
    //   touch_t {
    //     .selection = { {{ 100, 100, 0, 0, 50 }, { 100, 100, 0, 0, 50 }} },
    //     .castable = castable_t::add,
    //     .dtype = dtype
    //   },
    //   false);
    // tg.add_to_partial(b, b1,
    //   touch_t {
    //     .selection = { {{ 100, 100, 0, 0, 50 }, { 100, 100, 50, 50, 50 }} },
    //     .castable = castable_t::add,
    //     .dtype = dtype
    //   },
    //   false);
    // tg.add_to_partial(b, b1,
    //   touch_t {
    //     .selection = { {{ 100, 100, 50, 50, 50 }, { 100, 100, 0, 0, 100 }} },
    //     .castable = castable_t::add,
    //     .dtype = dtype
    //   },
    //   false);



    // // // case3: one inputs that can be simplified
    // // complete copy of b1 to b by doing first half then second half
    // int c1 = tg.insert_input(0, dtype, {50});
    // int c = tg.new_partial(0, dtype, {100});
    // tg.add_to_partial(c, c1,
    //   touch_t {
    //     .selection = { { 100, 100, 0, 0, 50 } },
    //     .castable = std::nullopt,
    //     .dtype = dtype
    //   },
    //   false);
    // tg.add_to_partial(c, c1,
    //   touch_t {
    //     .selection = { { 100, 100, 50, 50, 50 } },
    //     .castable = std::nullopt,
    //     .dtype = dtype
    //   },
    //   false);


    std::ofstream f("my_graph.gv");
    tg.print_graphviz(f);

    for(int id = 0; id != tg.nodes.size(); ++id) {
        auto const& node = tg.nodes[id];
        if(node.op.is_input()){
        auto const& [loc,size] = node.op.get_input();
        DOUT("input: " << id << ", with loc " << loc << ": " << size);
        }
        if(node.op.is_partialize()) {
            auto const& p = node.op.get_partialize();
            for(auto const& [inn, touch]: p.as_touches_from_flat()) {
                DOUT("partialize: " << id << ", with inn " << inn << ": " << touch);
            }
        }
    }
    tg.simplify_partializes();
    for(int id = 0; id != tg.nodes.size(); ++id) {
        auto const& node = tg.nodes[id];
        if(node.op.is_partialize()) {
            auto const& p = node.op.get_partialize();
            for(auto const& [inn, touch]: p.as_touches_from_flat()) {
                DOUT("simplified partialize: " << id << ", with inn " << inn << ": " << touch);
            }
        }
    }
}
