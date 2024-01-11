#include "../src/einsummable/reference.h"

#include "../src/einsummable/taskgraph.h"
#include <fstream>





// logic
// For two touch, (1)for one dim which ending index of one touch == starting index of another touch 
// AND (2)if starting and ending index of other dim are same for inputs and are same for outputs(input and output index does not have to be same) 
struct adj_simplify_partialize_t{
    // hold info that is needed to simply partialize
    int inn;
    touch_t first_touch;
    touch_t second_touch;
    touch_t simplify_touch;
};




touch_t merge_two_touch_dim(std::vector<touchdim_t> i_touch, std::vector<touchdim_t> j_touch, int merge_dim, touch_t i_touch_meta, touch_t j_touch_meta){
  if(i_touch[merge_dim].offset_inn + i_touch[merge_dim].size == j_touch[merge_dim].offset_inn && i_touch[merge_dim].offset_out + i_touch[merge_dim].size == j_touch[merge_dim].offset_out){
    touchdim_t new_touchdim = touchdim_t {
      .d_inn = i_touch[merge_dim].d_out,
      .d_out = i_touch[merge_dim].d_out,
      .offset_inn = i_touch[merge_dim].offset_inn,
      .offset_out = i_touch[merge_dim].offset_out,
      .size = i_touch[merge_dim].size + j_touch[merge_dim].size,
    };
    i_touch.erase(i_touch.begin() + merge_dim);
    i_touch.push_back(new_touchdim);
    return touch_t {
      .selection = i_touch,
      .castable = i_touch_meta.castable,
      .dtype = i_touch_meta.dtype
    };
  }
  else if (j_touch[merge_dim].offset_inn + j_touch[merge_dim].size == i_touch[merge_dim].offset_inn && j_touch[merge_dim].offset_out + j_touch[merge_dim].size == i_touch[merge_dim].offset_out){
    touchdim_t new_touchdim = touchdim_t {
      .d_inn = i_touch[merge_dim].d_out,
      .d_out = i_touch[merge_dim].d_out,
      .offset_inn = j_touch[merge_dim].offset_inn,
      .offset_out = j_touch[merge_dim].offset_out,
      .size = i_touch[merge_dim].size + j_touch[merge_dim].size,
    };
    i_touch.erase(i_touch.begin() + merge_dim);
    i_touch.push_back(new_touchdim);
    return touch_t {
      .selection = i_touch,
      .castable = i_touch_meta.castable,
      .dtype = i_touch_meta.dtype
    };
  }
  else{
    throw std::runtime_error("Don't know how to simplify two touches.");
  }
}

bool check_two_touchdim_t_same(touchdim_t a, touchdim_t b){
  return a.d_inn == b.d_inn && a.d_out == b.d_out && a.offset_inn == b.offset_inn && a.offset_out == b.offset_out && a.size == b.size;
}

bool check_two_touch_t_same(touch_t const& touch_a, touch_t const& touch_b){
  vector<touchdim_t> const& a = touch_a.selection;
  vector<touchdim_t> const& b = touch_b.selection;
  for(int k = 0; k < a.size(); k++){
    if(check_two_touchdim_t_same(a[k], b[k]) == false){
      return false;
    }
  }
  
  return true;
}

int check_two_touch_could_simplify(vector<touchdim_t> i_touch, vector<touchdim_t> j_touch){
  bool only_one_diff = false;
  int merge_dim = -1;
  for(int k = 0; k < i_touch.size(); k++){
    if(check_two_touchdim_t_same(i_touch[k], j_touch[k]) == false){
      if(only_one_diff == true) return -1;
      if ( i_touch[k].offset_inn + i_touch[k].size == j_touch[k].offset_inn && i_touch[k].offset_out + i_touch[k].size == j_touch[k].offset_out
          ||j_touch[k].offset_inn + j_touch[k].size == i_touch[k].offset_inn && j_touch[k].offset_out + j_touch[k].size == i_touch[k].offset_out){
            only_one_diff = true;
            merge_dim = k;
          }
    }
    
  }
  return merge_dim;
}

// find the simplified version of the current version
optional<adj_simplify_partialize_t> find_adj_simplify_partialize(std::map<int, std::vector<touch_t>> inn_touch_map){
  auto it = inn_touch_map.begin();
  while(it != inn_touch_map.end()){
    auto part_list = it->second;
    bool only_one_diff = false;
    if(part_list.size() > 1){
      // if there are more than one touch for the current input, check every pair of touch
      for(int i = 0; i < part_list.size(); ++i){
        vector<touchdim_t> i_touch = part_list[i].selection;
        for (int j = i+1; j < part_list.size(); ++j){
          vector<touchdim_t> j_touch = part_list[j].selection;
          int merge_dim = check_two_touch_could_simplify(i_touch, j_touch);
          if(merge_dim != -1){
            touch_t simplify_touch = merge_two_touch_dim(i_touch, j_touch, merge_dim, part_list[i], part_list[j]);
            return adj_simplify_partialize_t{
              .inn = it->first,
              .first_touch = part_list[i],
              .second_touch = part_list[j],
              .simplify_touch = simplify_touch
            };
          }
        }
      }
    }
    it++;
  }
  return std::nullopt;
}


taskgraph_t::partialize_t construct_partialize_from_map(std::map<int, std::vector<touch_t>> inn_touch_map){
  std::vector<tuple<int, touch_t>> touch_tuples;
  for (const auto& entry : inn_touch_map) {
    int key = entry.first;
    const std::vector<touch_t>& touch_vector = entry.second;

    // Iterate through touch_t elements in the vector and construct tuples
    for (const auto& touch : touch_vector) {
      touch_tuples.push_back(std::make_tuple(key, touch));
    }
  }



  return taskgraph_t::partialize_t::make_from_touches(0, touch_tuples);
}




// given a partialize and it's simplification, creates a new partialize that does the simplifcation.
void adj_simplify_partialize_(
  adj_simplify_partialize_t const& info,
  std::map<int, std::vector<touch_t>>& inn_touch_map){
    int key = info.inn;
    
    inn_touch_map[key].erase(std::remove_if(inn_touch_map[key].begin(), inn_touch_map[key].end(),
                    [&](touch_t const& elem) { return check_two_touch_t_same(elem, info.first_touch);}));
    
    inn_touch_map[key].erase(std::remove_if(inn_touch_map[key].begin(), inn_touch_map[key].end(),
                    [&](touch_t const& elem) { return check_two_touch_t_same(elem, info.second_touch); }));

    inn_touch_map[key].push_back(info.simplify_touch);
    
}


taskgraph_t::partialize_t adj_simplify_partialize(taskgraph_t::partialize_t const& partialize_) {
  taskgraph_t::partialize_t partialize = partialize_;
  std::map<int, std::vector<touch_t>> inn_touch_map;
  for(auto const& [inn, touch]: partialize.as_touches_from_flat()) {
    inn_touch_map[inn].push_back(touch);
  }

  auto maybe = find_adj_simplify_partialize(inn_touch_map);
  while(maybe.has_value()) {
    adj_simplify_partialize_(maybe.value(),inn_touch_map);
    maybe = find_adj_simplify_partialize(inn_touch_map);
  }
  auto ret = construct_partialize_from_map(inn_touch_map);
  return ret;
}


  
int main(){
    taskgraph_t tg;
    dtype_t dtype = default_dtype();

    // // case1: two different inputs for partialize
    // int a1 = tg.insert_input(0, dtype, {50});
    // int a2 = tg.insert_input(0, dtype, {50});
    // // int b  = tg.insert_input(0, dtype, {100});
    
    // int x = tg.new_partial(0, dtype, {100});
    // tg.add_to_partial(x, a1,
    //   touch_t {
    //     .selection = { { 50, 100, 0, 0, 50 } },
    //     .castable = std::nullopt,
    //     .dtype = dtype
    //   },
    //   false);
    // tg.add_to_partial(x, a2,
    //   touch_t {
    //     .selection = { { 50, 100, 0, 50, 50 } },
    //     .castable = std::nullopt,
    //     .dtype = dtype
    //   },
    //   false);
    
    // // verify using reference_partialize
    // dbuffer_t dataBufferA = make_dbuffer(dtype, 100);
    // dataBufferA.iota(0);
    // dbuffer_t dataBufferB = make_dbuffer(dtype, 100);
    // dataBufferB.iota(100);
    // map<int, dbuffer_t> inn_data{
    //   {0, dataBufferA},
    //   {1, dataBufferB}
    // };

    // auto const& part = tg.nodes[2].op.get_partialize();
    // dbuffer_t out = reference_partialize(part,inn_data);
    // DOUT("reference_partialize: " << out);

    
    


    // case2: one inputs that can not be simplified
    int b1 = tg.insert_input(0, dtype, {100,100});
    int b = tg.new_partial(0, dtype, {100,100});
    // two halves of b both copy first half of b1
    tg.add_to_partial(b, b1,
      touch_t {
        .selection = { {{ 100, 100, 0, 0, 50 }, { 100, 100, 0, 0, 50 }} },
        .castable = castable_t::add,
        .dtype = dtype
      },
      false);
    tg.add_to_partial(b, b1,
      touch_t {
        .selection = { {{ 100, 100, 0, 0, 50 }, { 100, 100, 50, 50, 50 }} },
        .castable = castable_t::add,
        .dtype = dtype
      },
      false);
    tg.add_to_partial(b, b1,
      touch_t {
        .selection = { {{ 100, 100, 50, 50, 50 }, { 100, 100, 0, 0, 100 }} },
        .castable = castable_t::add,
        .dtype = dtype
      },
      false);
  


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

            auto simplified_part = adj_simplify_partialize(p);
            for(auto const& [inn, touch]: simplified_part.as_touches_from_flat()) {
                DOUT("simplified partialize: " << id << ", with inn " << inn << ": " << touch);
            }
        }

    }
}
