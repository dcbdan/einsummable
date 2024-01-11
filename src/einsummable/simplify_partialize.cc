// #include "taskgraph.h"
// #include "graph.h"
// #include "touch.h"
// #include "memgraph.h"

// #include "../base/args.h"

// #include "../autoplace/apart.h"


// // logic
// // For two touch, (1)for one dim which ending index of one touch == starting index of another touch 
// // AND (2)if starting and ending index of other dim are same for inputs and are same for outputs(input and output index does not have to be same) 
// struct adj_simplify_partialize_t{
//     // hold info that is needed to simply partialize
//     int inn;
//     touch_t first_touch;
//     touch_t second_touch;
//     touch_t simplify_touch;
// };

// // find the simplified version of the current version
// optional<adj_simplify_partialize_t> find_adj_simplify_partialize(std::map<int, std::vector<touch_t>> inn_touch_map){
//   auto it = part_map.begin();
//   while(it != part_map.end()){
//     auto part_list = it->second;
//     bool only_one_diff = false;
//     if(part_list.size() > 1){
//       // if there are more than one touch for the current input, check every pair of touch
//       for(int i = 0; i < part_list.size(); ++i){
//         vector<touchdim_t> i_touch = part_list[i].selection;
//         for (int j = i; j < part_list.size(): ++j){
//           vector<touchdim_t> j_touch = part_list[j].selection;
//           int merge_dim = check_two_touch_could_simplify(i_touch, j_touch);
//           if(merge_dim != -1){
//             touch_t simplify_touch = merge_two_touch_dim(i_touch, j_touch, merge_dim, part_list[i], part_list[j]);
//             return adj_simplify_partialize_t{
//               .inn = it->first;
//               .first_touch = part_list[i];
//               .second_touch = part_list[j];
//               .simplify_touch = simplify_touch;
//             }
//           }
//         }
//       }
//     }
//     it = part_map.erase(it);
//   }
//   return std::nullopt;
// }


// touch_t merge_two_touch_dim(std::vector<touchdim_t> i_touch, std::vector<touchdim_t> j_touch, int merge_dim, touch_t i_touch_meta, touch_t j_touch_meta){
//   if(i_touch[merge_dim].offset_inn + i_touch[merge_dim].size == j_touch[merge_dim].offset_inn && i_touch[merge_dim].offset_out + i_touch[merge_dim].size == j_touch[merge_dim].offset_out){
//     touchdim_t new_touchdim = touchdim_t {
//       .d_inn = i_touch[merge_dim].d_out,
//       .d_out = i_touch[merge_dim].d_out,
//       .offset_inn = i_touch[merge_dim].offset_inn,
//       .offset_out = i_touch[merge_dim].offset_out,
//       .size = i_touch[merge_dim].size + j_touch[merge_dim].size,
//     };
//     i_touch.erase(i_touch.begin() + merge_dim);
//     i_touch.push_back(new_touchdim);
//     return touch_t {
//       .selection = i_touch,
//       .castable = i_touch_meta.castable,
//       .dtype = i_touch_meta.dtype
//     };
//   }
//   else if (j_touch[merge_dim].offset_inn + j_touch[merge_dim].size == i_touch[merge_dim].offset_inn && j_touch[merge_dim].offset_out + j_touch[merge_dim].size == i_touch[merge_dim].offset_out){
//     touchdim_t new_touchdim = touchdim_t {
//       .d_inn = i_touch[merge_dim].d_out,
//       .d_out = i_touch[merge_dim].d_out,
//       .offset_inn = j_touch[merge_dim].offset_inn,
//       .offset_out = j_touch[merge_dim].offset_out,
//       .size = i_touch[merge_dim].size + j_touch[merge_dim].size,
//     };
//     i_touch.erase(i_touch.begin() + merge_dim);
//     i_touch.push_back(new_touchdim);
//     return touch_t {
//       .selection = i_touch,
//       .castable = i_touch_meta.castable,
//       .dtype = i_touch_meta.dtype
//     };
//   }
//   else{
//     DOUT("Don't know how to simplify two touches.\n");
//   }

// }

// bool check_two_touchdim_t_same(touchdim_t a, touchdim_t b){
//   return a.d_inn == b.d_inn && a.d_out == b.d_out && a.offset_inn == b.offset_inn && a.offset_out == b.offset_out && a.size == b.size;
// }

// bool check_two_vector_touch_same(std::vector<touchdim_t> a, std::vector<touchdim_t> b){
//   for(int k = 0; k < a.size(); k++){
//     if(check_two_touchdim_t_same(a[k], b[k]) == false){
//       return false;
//     }
//   }
//   return true;
// }

// int check_two_touch_could_simplify(vector<touchdim_t> i_touch, vector<touchdim_t> j_touch){
//   bool only_one_diff = false;
//   int merge_dim = -1;
//   for(int k = 0; k < i_touch.size(); k++){
//     if(check_two_touchdim_t_same(i_touch[k], j_touch[k]) == false){
//       if(only_one_diff == true) return -1;
//       if ( i_touch[merge_dim].offset_inn + i_touch[merge_dim].size == j_touch[merge_dim].offset_inn && i_touch[merge_dim].offset_out + i_touch[merge_dim].size == j_touch[merge_dim].offset_out
//           ||j_touch[merge_dim].offset_inn + j_touch[merge_dim].size == i_touch[merge_dim].offset_inn && j_touch[merge_dim].offset_out + j_touch[merge_dim].size == i_touch[merge_dim].offset_out){
//             only_one_diff = true;
//             merge_dim = k;
//           }
//     }
//   }
//   return merge_dim;
// }



// // given a partialize and it's simplification, creates a new partialize that does the simplifcation.
// void adj_simplify_partialize_(
//   adj_simplify_partialize_t const& info,
//   std::map<int, std::vector<touch_t>> inn_touch_map){
//     int key = info.inn;
//     inn_touch_map[key].erase(
//         std::remove_if(inn_touch_map[key].begin(), inn_touch_map[key].end(),
//                        [](vector<touchdim_t> elem, adj_simplify_partialize_t const& info) { return check_two_vector_touch_same(elem, info.first_touch.selection); }),
//         inn_touch_map[key].end());

//     inn_touch_map[key].erase(
//         std::remove_if(inn_touch_map[key].begin(), inn_touch_map[key].end(),
//                        [](vector<touchdim_t> elem, adj_simplify_partialize_t const& info) { return check_two_vector_touch_same(elem, info.second_touch.selection); }),
//         inn_touch_map[key].end());

//     inn_touch_map[key].push_back(info.simplify_touch);
    
//   }





