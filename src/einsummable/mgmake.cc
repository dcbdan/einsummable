#include "mgmake.h"

tuple<map<int, mem_t>, // input -> mem
      map<int, mem_t>, // save -> mem
      memgraph_t>
memgraph_t::make_without_evict(taskgraph_t const&   taskgraph,
                               vector<uint64_t>     mem_sizes,
                               allocator_settings_t settings)
{
    int const n_compute_locs = taskgraph.num_locs();

    vector<int> which_storage(n_compute_locs);
    std::iota(which_storage.begin(), which_storage.end(), 0);

    auto [inn_to_memdata, save_to_memdata, memgraph] =
        make(taskgraph, which_storage, mem_sizes, {}, settings, false);

    map<int, mem_t> inn_to_mem;
    for (auto const& [tid, memdata] : inn_to_memdata) {
        inn_to_mem.insert({tid, memdata.get_memloc().as_mem()});
    }

    map<int, mem_t> save_to_mem;
    for (auto const& [tid, memdata] : save_to_memdata) {
        save_to_mem.insert({tid, memdata.get_memloc().as_mem()});
    }

    return {inn_to_mem, save_to_mem, memgraph};
}

tuple<map<int, memstoloc_t>, map<int, memstoloc_t>, memgraph_t>
memgraph_t::make(taskgraph_t const&    graph,
                 vector<int>           which_storage,
                 vector<uint64_t>      mem_sizes,
                 map<int, memstoloc_t> init_input_tid_to_data,
                 allocator_settings_t  settings,
                 bool                  use_storage)
{
    auto&& [i, o, a_nullopt, m] = make_(
        graph, which_storage, mem_sizes, init_input_tid_to_data, settings, use_storage, false);
    return {std::move(i), std::move(o), std::move(m)};
}

memgraph_t memgraph_make_state_t::pop_memgraph()
{
    if (partializes_in_progress.size() > 0) {
        throw std::runtime_error("partialize in progress; cannot pop memgraph");
    }

    memgraph_t ret = memgraph;

    memgraph =
        memgraph_t(ret.num_compute_locs, ret.num_storage_locs, ret.storage_locs, ret.prune_edges);

    // update the trio
    //   task_tensor_to_mem_node
    //   tensors_on_storage;
    //   tensors_on_memory;
    //
    // also update the allocators to not depend on the ret mid's

    for (auto iter = task_tensor_to_mem_node.begin(); iter != task_tensor_to_mem_node.end();
         ++iter) {
        int const& tid = iter->first;

        int         ret_mid = iter->second;
        auto const& ret_op = ret.nodes[ret_mid].op;

        int& new_mid = iter->second;
        if (tensors_on_storage.count(tid) > 0) {
            // insert on storage
            // ret op must either be a inputsto itself or a
            // an evict node
            inputsto_t sto;
            if (ret_op.is_inputsto()) {
                sto = ret_op.get_inputsto();
            } else if (ret_op.is_evict()) {
                auto const& evict = ret_op.get_evict();
                sto.storage_loc = evict.dst.loc;
                sto.storage_id = evict.dst.id;
                sto.size = evict.src.size;
            }

            new_mid = memgraph.insert(op_t(sto), set<int>{});

            // tensors_on_storage does not change
        } else {
            // insert on memory
            inputmem_t mem = inputmem_t::from_memloc(ret_op.get_output_memloc());
            new_mid = memgraph.insert(op_t(mem), set<int>{});

            // this tid has not been used by any mids now
            tensors_on_memory.at(tid) = set<int>{};
        }
    }

    for (allocator_t& allocator : allocators) {
        allocator.clear_dependencies();
    }

    // These should not be accessed again for what it contained
    task_node_to_mem_node = {};
    task_touch_to_mem_node = {};

    return ret;
}

tuple<map<int, memstoloc_t>, // input -> data
      map<int, memstoloc_t>, // save  -> data
      optional<memgraph_t>,
      memgraph_t>
memgraph_t::make_(taskgraph_t const&    taskgraph,
                  vector<int>           which_storage,
                  vector<uint64_t>      mem_sizes,
                  map<int, memstoloc_t> input_tid_to_data,
                  allocator_settings_t  settings,
                  bool                  use_storage,
                  bool                  split_off_inputs)
{
    int n_compute_locs = taskgraph.num_locs();
    if (mem_sizes.size() > n_compute_locs) {
        n_compute_locs = mem_sizes.size();
    }

    if (which_storage.size() == 0) {
        which_storage = vector<int>(n_compute_locs);
        std::iota(which_storage.begin(), which_storage.end(), 0);
    }

    if (which_storage.size() != n_compute_locs) {
        throw std::runtime_error("incorrect which storage length: memgraph_t::make");
    }

    int n_storage_locs = 0;
    for (int const& storage_loc : which_storage) {
        if (storage_loc < 0) {
            throw std::runtime_error("invalid storage loc");
        }
        n_storage_locs = std::max(n_storage_locs, storage_loc + 1);
    }

    for (int i = 0; i != n_storage_locs; ++i) {
        auto iter = std::find(which_storage.begin(), which_storage.end(), i);
        if (iter == which_storage.end()) {
            throw std::runtime_error("storage locs must have 0, ..., n_storage_locs-1; no missing");
        }
    }

    vector<allocator_t> allocators;
    if (mem_sizes.size() == 0) {
        // set up an allocator for each loc,
        // each with a very large amount of available memory
        allocators = vector<allocator_t>(
            n_compute_locs, allocator_t(std::numeric_limits<uint64_t>::max(), settings));
    } else {
        if (mem_sizes.size() != n_compute_locs) {
            throw std::runtime_error("must have mem sizes for each device");
        }
        allocators.reserve(mem_sizes.size());
        for (auto const& m : mem_sizes) {
            allocators.emplace_back(m, settings);
        }
    }

    memgraph_make_state_t state(taskgraph,
                                which_storage,
                                allocators,
                                input_tid_to_data,
                                n_compute_locs,
                                n_storage_locs,
                                use_storage);

    if (!use_storage) {
        // Without storage, it makes more sense to allocate all the inputs
        // before proceeding
        for (int id = 0; id != taskgraph.nodes.size(); ++id) {
            auto const& node = taskgraph.nodes[id];
            if (node.op.is_input()) {
                if (node.outs.size() == 0 && !node.is_save) {
                    throw std::runtime_error(
                        "This is goofy: an input to memgraph is not used or saved."
                        " Call this again after pruning inputs that don't get used"
                        " or saved.");
                }

                // It could be the case that the used initialized the input
                if (!state.input_has_been_initialized(id)) {
                    state.initialize_input(id);
                }
            }
        }
    }

    optional<memgraph_t> input_memgraph;
    if (split_off_inputs) {
        auto [input_tg_ops, core_tg_ops] = order_split_taskgraph(taskgraph);
        state.process(input_tg_ops);
        input_memgraph = state.pop_memgraph();
        state.process(core_tg_ops);
    } else {
        state.process(order_taskgraph(taskgraph));
    }

    map<int, memstoloc_t> save_to_data;
    for (int id = 0; id != taskgraph.nodes.size(); ++id) {
        auto const& node = taskgraph.nodes[id];
        if (node.is_save) {
            int         memid = state.task_tensor_to_mem_node.at(id);
            auto const& memnode = state.memgraph.nodes[memid];
            save_to_data.insert({id, memnode.op.get_output_memstoloc()});
        }
    }
    std::cout << "Total nodes being saved: " << save_to_data.size() << std::endl;

    return {input_tid_to_data, save_to_data, input_memgraph, state.memgraph};
}

vector<_which_op_t> build_tg_ops(taskgraph_t const& taskgraph, vector<int> const& tids_in_order)
{
    vector<_which_op_t> ret;
    ret.reserve(2 * taskgraph.nodes.size());

    for (auto const& id : tids_in_order) {
        auto const& node = taskgraph.nodes[id];
        if (node.op.is_input()) {
            // Input nodes have already been provided
        } else if (node.op.is_partialize()) {
            auto const& p = node.op.get_partialize();
            for (int which_unit = 0; which_unit != p.units.size(); ++which_unit) {
                auto const& unit = p.units[which_unit];
                for (int which_inn = 0; which_inn != unit.inputs.size(); ++which_inn) {
                    ret.emplace_back(_which_touch_t{
                        .task_id = id, .unit_id = which_unit, .touch_id = which_inn});
                }
            }
        } else {
            // apply or move
            ret.emplace_back(_which_node_t{.task_id = id});
        }
    }

    return ret;
}

vector<_which_op_t> order_taskgraph(taskgraph_t const& taskgraph)
{
    return build_tg_ops(taskgraph, taskgraph.get_order());
}

tuple<vector<_which_op_t>, vector<_which_op_t>> order_split_taskgraph(taskgraph_t const& taskgraph)
{
    auto [inn_order, core_order] = taskgraph.get_input_core_order();
    return {build_tg_ops(taskgraph, inn_order), build_tg_ops(taskgraph, core_order)};
}

vector<tuple<int, _which_touch_t>> get_which_touches_from(taskgraph_t const& taskgraph, int out)
{
    vector<tuple<int, _which_touch_t>> ret;

    auto const& out_node = taskgraph.nodes[out];
    if (!out_node.op.is_partialize()) {
        throw std::runtime_error("invalid call to get_which_touches_from");
    }
    // get the partialize.
    // partialize consists of units, units consist of input ops.
    // Figure out if each input op of each unit matches to this
    //   tensor just formed. If so, add it.
    auto const& partialize = out_node.op.get_partialize();
    for (int which_unit = 0; which_unit != partialize.units.size(); ++which_unit) {
        auto const& unit = partialize.units[which_unit];
        for (int which_inn = 0; which_inn != unit.inputs.size(); ++which_inn) {
            auto const& inn = unit.inputs[which_inn].id;
            ret.push_back(
                {inn,
                 _which_touch_t{.task_id = out, .unit_id = which_unit, .touch_id = which_inn}});
        }
    }

    return ret;
}

vector<_which_touch_t> get_which_touches_from_to(taskgraph_t const& tg, int out, int inn)
{
    vector<_which_touch_t> ret;
    for (auto const& [inn_, w] : get_which_touches_from(tg, out)) {
        if (inn == inn_) {
            ret.push_back(w);
        }
    }
    return ret;
}

memgraph_make_state_t::memgraph_make_state_t(taskgraph_t const&         tg,
                                             vector<int> const&         which_storage,
                                             vector<allocator_t> const& as,
                                             map<int, memstoloc_t>&     ittd,
                                             int                        num_compute,
                                             int                        num_storage,
                                             bool                       use_storage)
    : taskgraph(tg),
      memgraph(num_compute, num_storage, which_storage),
      allocators(as),
      _group(0),
      use_storage(use_storage),
      input_tid_to_data(ittd)
{
#ifdef USE_LOCATIONWISE_APPLY_ORDERING
    last_applys = vector<int>(num_compute, -1);
    DOUT("Using locationwise apply ordering");
#endif
    _sto_id = 0;
    remaining_usage_counts = vector<int>(taskgraph.nodes.size(), 0);

    for (auto const& allocator : allocators) {
        if (!allocator.is_empty()) {
            throw std::runtime_error("initial allocators to memgraph_make_state_t should be empty");
        }
    }

    // - tell the allocators what memory is being used at time zero
    // - update sto_id accordingly
    // - insert onto task_tensor_to_mem_node
    for (auto const& [tid, memstoloc] : input_tid_to_data) {
        if (memstoloc.is_memloc()) {
            auto const& [offset, size, loc] = memstoloc.get_memloc();
            if (!allocators.at(loc).allocate_at_without_deps(offset, size)) {
                throw std::runtime_error("could not allocate at previously alloced location");
            }

            inputmem_t input{.loc = loc, .offset = offset, .size = size};

            int mid = memgraph.insert(op_t(input), {});
            task_tensor_to_mem_node_insert_on_memory(tid, mid);
        } else if (memstoloc.is_stoloc()) {
            // Note that even though storage ids are specific to storage locs,
            // this code only cares about finding a unique storage id whenever
            // one is needed.
            auto const& [storage_loc, storage_id] = memstoloc.get_stoloc();
            _sto_id = std::max(_sto_id, 1 + storage_id);

            inputsto_t input{
                .storage_loc = storage_loc, .storage_id = storage_id, .size = tg.out_size(tid)};

            int mid = memgraph.insert(op_t(input), {});
            task_tensor_to_mem_node_insert_on_storage(tid, mid);
        }
    }

    // We may have an einsummable y = x + x. In this case,
    // x gets used once by y.
    //
    // We may also have  a partialize
    //   y = touch from the left  side of x
    //   y = touch from the right side of x
    // In this case, x gets used twice in the formation of y,
    // once for each touch.

    for (int id = 0; id != taskgraph.nodes.size(); ++id) {
        auto const& node = taskgraph.nodes[id];
        if (node.op.is_partialize()) {
            auto const& partialize = node.op.get_partialize();
            // each touch incurs a usage, even if there are multiple touches
            // from the sasme input
            for (auto const& [inn, _] : partialize.as_touches_from_flat()) {
                remaining_usage_counts[inn] += 1;
            }
        } else {
            set<int> inns = node.op.inputs();
            // for input nodes, inns is empty
            // for move nodes, there is only one input
            // for apply nodes, we can't double count like x in y = x + x,
            //   so inns being a set is desired
            for (auto const& inn : inns) {
                remaining_usage_counts[inn] += 1;
            }
        }
    }
}

void memgraph_make_state_t::initialize_input(int inn)
{
    auto const& node = taskgraph.nodes[inn];
    int         loc = node.op.out_loc();
    uint64_t    size = node.op.out_size();

    auto maybe = allocators.at(loc).allocate_without_deps(size);
    if (maybe) {
        // If we are able to allocate without deps on memory, insert a inputmem_t
        auto const& offset = maybe.value();

        inputmem_t input_mem = {.loc = loc, .offset = offset, .size = size};
        input_tid_to_data[inn] = memstoloc_t(input_mem.as_memloc());

        op_t input_op = op_t(input_mem);
        int  memid = memgraph.insert(input_op, {});

        task_tensor_to_mem_node_insert_on_memory(inn, memid);
    } else {
        // If we are not able to allocate on memory, insert into inputsto_t

        if (!use_storage) {
            throw std::runtime_error("no more memory to initialize inputs; use storage?");
        }

        inputsto_t input_sto = {
            .storage_loc = memgraph.storage_locs[loc], .storage_id = _sto_id++, .size = size};
        input_tid_to_data[inn] = memstoloc_t(input_sto.as_stoloc());

        op_t input_op = op_t(input_sto);
        int  memid = memgraph.insert(input_op, {});
        task_tensor_to_mem_node_insert_on_storage(inn, memid);
    }
}

bool memgraph_make_state_t::input_has_been_initialized(int inn)
{
    return input_tid_to_data.find(inn) != input_tid_to_data.end();
}

bool memgraph_make_state_t::allocate_op(_which_op_t const& which_op, bool force)
{
    int id;

    if (std::holds_alternative<_which_node_t>(which_op)) {
        id = std::get<_which_node_t>(which_op).task_id;
    } else {
        id = std::get<_which_touch_t>(which_op).task_id;
    }

    auto const& node = taskgraph.nodes[id];

    // Note that input_t nodes are not encoded in _which_ops
    // and they may or may not already have been added to the memgraph.
    //
    // Make sure to add any input_t nodes before proceeding,
    // otherwise uninitialized input_t nodes will be treated as
    // new tensors below and would get mistakenly allocated.
    for (int const& inn : node.op.inputs()) {
        auto const& inn_node = taskgraph.nodes[inn];
        if (inn_node.op.is_input() && !input_has_been_initialized(inn)) {
            initialize_input(inn);
        }
    }

    // Set used_tids
    vector<int> used_tids;
    if (node.op.is_apply()) {
        auto const& [loc, inns, es] = node.op.get_apply();

        vector<int>& inns_then_out = used_tids;
        inns_then_out = vector<int>(inns.size() + 1);
        std::copy(inns.begin(), inns.end(), inns_then_out.begin());
        inns_then_out[inns.size()] = id;

        used_tids = inns_then_out;
    } else if (node.op.is_constant()) {
        auto const& constant = node.op.get_constant();
        auto const& fill = constant.fill;

        used_tids.push_back(id);
    } else if (node.op.is_move()) {
        auto const& [src, dst, task_inn, size] = node.op.get_move();
        used_tids = {task_inn, id};
    } else if (node.op.is_partialize()) {
        auto const& partialize = node.op.get_partialize();

        auto const& [_0, unit_id, touch_id] = std::get<_which_touch_t>(which_op);
        auto [task_inn, touch] = partialize.get_touch(unit_id, touch_id);

        used_tids = {task_inn, id};
    } else {
        throw std::runtime_error("should not reach");
    }

    if (force) {
        force_allocate_tids(used_tids);
        return true;
    } else {
        return allocate_tids_without_evict(used_tids);
    }
}

bool memgraph_make_state_t::allocate_tids_without_evict(vector<int> const& used_tids)
{
    // Note: it may be the case that these tids do not all belong to the
    //       same memory location!

    // All the save states
    vector<allocator_t::save_t> saves(allocators.size());
    bool                        failed = false;

    // A quick wrapper to only form saves as needed;;
    // doing this prevents us from calling the naive save
    // for every allocator
    auto do_allocate = [&](int loc, uint64_t size) {
        auto& allocator = allocators.at(loc);
        auto& save = saves.at(loc);
        if (!save) {
            save = allocator.checkpoint();
        }
        auto ret = allocator.allocate(size);
        return ret;
    };

    vector<int>                       tids_need_alloc;
    vector<tuple<uint64_t, set<int>>> allocs;
    for (int const& tid : used_tids) {
        auto const& node = taskgraph.nodes[tid];
        auto        iter = task_tensor_to_mem_node.find(tid);
        if (iter != task_tensor_to_mem_node.end()) {
            // We have this tensor, but it may be on storage,
            // so move it to memory
            int const& memid = iter->second;
            auto       maybe_mem = memgraph.nodes[memid].op.get_output_memstoloc();
            if (maybe_mem.is_stoloc()) {
                auto maybe = do_allocate(node.op.out_loc(), node.op.out_size());
                if (!maybe) {
                    failed = true;
                    break;
                } else {
                    tids_need_alloc.push_back(tid);
                    allocs.push_back(maybe.value());
                }
            } else {
                // that's fine, we don't have to allocate it then
            }
        } else {
            // This is an output tid that needs to be allocated
            if (node.op.is_input()) {
                throw std::runtime_error(
                    "The input node must already be in task_tensor_to_mem_node!");
            }
            auto maybe = do_allocate(node.op.out_loc(), node.op.out_size());
            if (!maybe) {
                failed = true;
                break;
            } else {
                tids_need_alloc.push_back(tid);
                allocs.push_back(maybe.value());
            }
        }
    }

    if (failed) {
        for (int loc = 0; loc != allocators.size(); ++loc) {
            auto& save = saves[loc];
            if (save) {
                allocators.at(loc).reset(save);
            }
        }
        return false;
    } else {
        // clear up the saves
        saves.resize(0);
    }

    vector<int> alloc_mids;
    alloc_mids.reserve(allocs.size());
    for (auto const& [tid, info] : vector_zip(tids_need_alloc, allocs)) {
        int      loc = taskgraph.out_loc(tid);
        uint64_t size = taskgraph.out_size(tid);
        auto const& [offset, deps] = info;
        alloc_t alloc{.loc = loc, .offset = offset, .size = size};
        int     mid = memgraph.insert(op_t(alloc), deps);
        alloc_mids.push_back(mid);
    }

    for (auto const& [tid, alloc_mid] : vector_zip(tids_need_alloc, alloc_mids)) {
        auto iter = task_tensor_to_mem_node.find(tid);
        if (iter != task_tensor_to_mem_node.end()) {
            // In this case, the tensor was on storage, so load it into
            // the memory at alloc_mid
            _load_tensor_helper(tid, alloc_mid);
        } else {
            // This mustve been a new tensor, so get it setup
            task_tensor_to_mem_node_insert_on_memory(tid, alloc_mid);
        }
    }

    return true;
}

void memgraph_make_state_t::force_allocate_tids(vector<int> const& tids)
{
    for (int const& tid : tids) {
        auto const& node = taskgraph.nodes[tid];
        auto        iter = task_tensor_to_mem_node.find(tid);
        if (iter != task_tensor_to_mem_node.end()) {
            // We have this tensor, but it may be on storage,
            // so move it to memory
            int const& memid = iter->second;
            auto       maybe_mem = memgraph.nodes[memid].op.get_output_memstoloc();
            if (maybe_mem.is_stoloc()) {
                // Note: we are not pushing other tids in tids off of memory
                load_tensor_with_evict(tid, tids);
                int const& memid = task_tensor_to_mem_node.at(tid);
            }
        } else {
            // This is an output tid that needs to be allocated
            if (node.op.is_input()) {
                throw std::runtime_error(
                    "The input node must already be in task_tensor_to_mem_node!");
            }

            // See to it that the memory gets allocated, possibly with evictions.
            int      loc = node.op.out_loc();
            uint64_t size = node.op.out_size();
            int      alloc_mid = allocate_with_evict(loc, size, tids);
            mem_t    alloc_mem = memgraph.nodes[alloc_mid].op.get_alloc().as_mem();
            // make sure to add the memid into task_tensor_to_mem_node
            // so we don't keep allocating this output memory!
            task_tensor_to_mem_node_insert_on_memory(tid, alloc_mid);
        }
    }
}

bool memgraph_make_state_t::add_op(_which_op_t const& which_op)
{
    int id;
    if (std::holds_alternative<_which_node_t>(which_op)) {
        id = std::get<_which_node_t>(which_op).task_id;
    } else {
        id = std::get<_which_touch_t>(which_op).task_id;
    }
    // std::cout << "add_op with id: " << id << std::endl;

    auto const& node = taskgraph.nodes[id];

    // These are the tensors that will get marked as used
    // and possibly deleted once we're done with the op
    set<int> used_task_tensors;
    // For apply nodes, the set of input tensors
    // For a single touch out of a partialize, the input tensor

    // These are the tensors whose corresponding memgraph node
    // is getting used by the node that will be created
    vector<int> used_tids;

    // Note that used_task_tensors and used_tids are doing
    // different things. used_tids is for updating tensors_on_memory
    // and used_task_tensors is for register_usage.

    // This is the op, deps we're going to form that'll
    // create a new node in the memgraph.
    optional<op_t> op;
    set<int>       deps;

    // For bookkeeping with a new touch
    optional<int> touch_output_memid;

    if (node.op.is_apply()) {
        // DOUT("addop: is apply");
        auto const& [loc, inns, es] = node.op.get_apply();

        vector<int>& out_then_inns = used_tids;
        out_then_inns = vector<int>(inns.size() + 1);
        out_then_inns[0] = id;
        std::copy(inns.begin(), inns.end(), out_then_inns.begin() + 1);

        used_tids = out_then_inns;

        auto [vector_deps, mems] = vector_unzip(get_tensors_in_memory_without_alloc(out_then_inns));
        deps = set<int>(vector_deps.begin(), vector_deps.end());

#ifdef USE_LOCATIONWISE_APPLY_ORDERING
        {
            // Add the last apply on this gpu
            int last_apply_mid = last_applys[loc];
            if (last_apply_mid >= 0) {
                deps.insert(last_apply_mid);
            }
        }
#endif

        for (auto const& task_inn : inns) {
            used_task_tensors.insert(task_inn);
        }

        op = op_t(apply_t{.loc = loc, .mems = mems, .op = es, .group = -1});
    } else if (node.op.is_constant()) {
        // DOUT("addop: is constant");
        auto const& constant = node.op.get_constant();
        auto const& fill = constant.fill;

        used_tids.push_back(id);
        auto [vector_deps, mems] = vector_unzip(get_tensors_in_memory_without_alloc(used_tids));
        deps = set<int>(vector_deps.begin(), vector_deps.end());
        auto const& mem = mems[0];

        op = op_t(constant_t{.loc = constant.loc, .offset = mem.offset, .fill = constant.fill});
    } else if (node.op.is_move()) {
        // DOUT("addop: is move");
        auto const& [src, dst, task_inn, size] = node.op.get_move();

        used_tids = {task_inn, id};
        auto info = get_tensors_in_memory_without_alloc(used_tids);
        auto const& [src_mem_id, src_mem] = info[0];
        auto const& [dst_mem_id, dst_mem] = info[1];

        deps.insert(src_mem_id);
        deps.insert(dst_mem_id);

        used_task_tensors.insert(task_inn);

        op = op_t(move_t{.src = {src, src_mem.offset}, .dst = {dst, dst_mem.offset}, .size = size});
    } else if (node.op.is_partialize()) {
        // DOUT("addop: is partialize");
        auto const& partialize = node.op.get_partialize();

        auto const& [_0, unit_id, touch_id] = std::get<_which_touch_t>(which_op);
        auto [task_inn, touch] = partialize.get_touch(unit_id, touch_id);

        used_tids = {task_inn, id};
        auto info = get_tensors_in_memory_without_alloc(used_tids);
        auto const& [inn_mem_id, inn_mem] = info[0];
        auto const& [out_mem_id, out_mem] = info[1];

        touch_output_memid = out_mem_id;

        deps.insert(inn_mem_id);
        deps.insert(out_mem_id);

        used_task_tensors.insert(task_inn);

        op = op_t(apply_t{.loc = partialize.loc,
                          .mems = {out_mem, inn_mem},
                          .op = touch,
                          .group = get_group_at(id, unit_id)});
    } else {
        throw std::runtime_error("should not reach");
    }

    int new_memid = memgraph.insert(op.value(), deps);

#ifdef USE_LOCATIONWISE_APPLY_ORDERING
    {
        if (node.op.is_apply()) {
            auto const& [loc, inns, es] = node.op.get_apply();
            last_applys[loc] = new_memid;
        }
    }
#endif

    // notify tensors_on_memory that these tensors were used
    for (auto const& used_tid : used_tids) {
        tensors_on_memory[used_tid].insert(new_memid);
    }

    if (std::holds_alternative<_which_node_t>(which_op)) {
        task_node_to_mem_node.insert({std::get<_which_node_t>(which_op), new_memid});

        // For apply and move nodes, insert the newly created
        // memid into the tensor mapping
        task_tensor_to_mem_node_update_on_memory(id, new_memid);
    } else {
        // This is a touch in a partialize node.
        task_touch_to_mem_node.insert({std::get<_which_touch_t>(which_op), new_memid});
        int num_touches_in_partialize = node.op.get_partialize().get_num_touches();

        auto& in_progress = partializes_in_progress[id];
        in_progress.push_back(new_memid);

        bool is_last_touch = (in_progress.size() == num_touches_in_partialize);
        if (is_last_touch) {
            // This partialize is complete
            if (num_touches_in_partialize == 1) {
                // then insert the newly created memid into the tensor mapping
                task_tensor_to_mem_node_update_on_memory(id, new_memid);
            } else {
                // create a partialize node that depends on everything in in_progress
                // and insert that into the tensor mapping
                auto         new_memloc = memgraph.nodes[new_memid].op.get_output_memloc();
                partialize_t new_partialize = partialize_t::from_memloc(new_memloc);
                int          partialize_memid = memgraph.insert(
                    op_t(new_partialize), set<int>(in_progress.begin(), in_progress.end()));
                task_tensor_to_mem_node_update_on_memory(id, partialize_memid);
            }
            partializes_in_progress.erase(id);
        } else {
            // This partialize is still in progress. Insert the allocated
            // output memid.
            task_tensor_to_mem_node_update_on_memory(id, touch_output_memid.value());
        }
    }

    // Now try to delete some tensors
    bool just_deleted = false;
    for (auto const& used_task_id : used_task_tensors) {
        bool has_delete = register_usage(used_task_id);
        just_deleted = just_deleted || has_delete;
    }

    return just_deleted;
}

void memgraph_make_state_t::process(vector<_which_op_t> const& all_ops)
{
    // Since storage may be used, setup a structure containing info on
    // when something will be used next.
    {
        vector<set<int>> usage(taskgraph.nodes.size());
        for (int oid = 0; oid != all_ops.size(); ++oid) {
            auto const& which_op = all_ops[oid];
            if (std::holds_alternative<_which_node_t>(which_op)) {
                auto const& [task_id] = std::get<_which_node_t>(which_op);
                usage[task_id].insert(oid);
                for (auto const& inn_id : taskgraph.nodes[task_id].op.inputs()) {
                    usage[inn_id].insert(oid);
                }
            } else {
                auto const& [task_id, unit_id, touch_id] = std::get<_which_touch_t>(which_op);
                usage[task_id].insert(oid);
                auto const& p = taskgraph.nodes[task_id].op.get_partialize();
                int const&  inn_id = p.units[unit_id].inputs[touch_id].id;
                usage[inn_id].insert(oid);
            }
        }

        order_state = order_state_t{.when_used = usage, .threshold = 0};
    }

    // Do each op, updating ostate threshold so that items can be compared
    // based on when they'll be used next
    auto& ostate = order_state.value();
    int   alloc_oid = 0;
    int   done_oid = 0;
    bool  do_alloc = false; // if we deleted any tensor in the last iteration
    while (done_oid < all_ops.size()) {
        ostate.threshold = done_oid;
        if (do_alloc) {
            // Allocate as many nodes as we can
            while (alloc_oid < all_ops.size() && allocate_op(all_ops.at(alloc_oid), false)) {
                alloc_oid++;
            }
        }
        if (alloc_oid == done_oid) {
            // make sure that the op we're about to do is allocated
            allocate_op(all_ops.at(alloc_oid), true);
            alloc_oid++;
        }

        // Do the corresponding op. If anything got deleted,
        // do_alloc will be set to true
        do_alloc = add_op(all_ops.at(done_oid));
        done_oid++;
    }
}

int memgraph_make_state_t::get_group_at(int task_id, int unit_id)
{
    auto const& node = taskgraph.nodes[task_id];
    auto const& partialize = node.op.get_partialize();
    auto const& unit = partialize.units[unit_id];
    if (unit.inputs.size() == 1) {
        return -1;
    } else {
        if (to_group.count({task_id, unit_id}) == 0) {
            to_group.insert({{task_id, unit_id}, _group});
            _group += 1;
        }
        return to_group.at({task_id, unit_id});
    }
}

// Helper to print out what is inside the mapping
// task_node_to_mem_node(task apply/move node to memid)
void memgraph_make_state_t::print_task_node_to_mem_node(
    map<_which_node_t, int> task_node_to_mem_node)
{
    DOUT("Printing task_node_to_mem_node:");
    for (auto iter = task_node_to_mem_node.begin(); iter != task_node_to_mem_node.end(); ++iter) {
        std::cout << "apply/move id: " << iter->first.task_id << "; memid: " << iter->second;
    }
    std::cout << std::endl;
}

void memgraph_make_state_t::print_task_touch_to_mem_node(
    map<_which_touch_t, int> task_touch_to_mem_node)
{
    DOUT("Printing task_touch_to_mem_node:");
    for (auto iter = task_touch_to_mem_node.begin(); iter != task_touch_to_mem_node.end(); ++iter) {
        std::cout << "partialize task id: " << iter->first.task_id
                  << "unit id: " << iter->first.unit_id << "touch id: " << iter->first.touch_id
                  << "; memid:" << iter->second;
    }
    std::cout << std::endl;
}

bool memgraph_make_state_t::register_usage(int task_id)
{
    int& rem = remaining_usage_counts[task_id];
    if (rem == 0) {
        throw std::runtime_error("this should not happen");
    }
    rem -= 1;
    if (rem == 0) {
        if (donated.count(task_id) > 0) {
            // can't be deleted since this guy was donated to where it was used
            donated.erase(task_id);
            return false;
        }

        auto const& node = taskgraph.nodes[task_id];

        if (node.is_save) {
            // can't be deleted since it is marked for saving
            return false;
        }

        int         completing_memnode = task_tensor_to_mem_node.at(task_id);
        auto const& memnode = memgraph.nodes[completing_memnode];
        memstoloc_t data = memnode.op.get_output_memstoloc();

        if (data.is_stoloc()) {
            // TODO: this could get triggered if we're accomplishing a
            //       taskgraph move via a memgraph load
            throw std::runtime_error("this tensor should have been deleted, not evicted!");
        }

        memloc_t memloc = data.get_memloc();
        del_t    del = del_t::from_memloc(memloc);

        // The delete of task_id depends on
        // 1. the output applys that used task_id and
        // 2. the touch operations that used task_id
        set<int> del_deps;
        for (int const& task_out : node.outs) {
            auto const& out_node = taskgraph.nodes[task_out];
            if (out_node.op.is_apply() || out_node.op.is_move()) {
                _which_node_t _task_out{task_out};
                auto          iter = task_node_to_mem_node.find(_task_out);
                if (iter == task_node_to_mem_node.end()) {
                    // Assumption: this _task_out lived on the other side of a pop_memgraph
                } else {
                    del_deps.insert(iter->second);
                }
            } else if (out_node.op.is_partialize()) {
                auto const whiches = get_which_touches_from_to(taskgraph, task_out, task_id);
                for (auto const& which : whiches) {
                    auto iter = task_touch_to_mem_node.find(which);
                    if (iter == task_touch_to_mem_node.end()) {
                        // Assumption: this which lived on the other side of a pop_memgraph
                    } else {
                        del_deps.insert(iter->second);
                    }
                }
            }
        }

        int del_id = memgraph.insert(op_t(del), del_deps);

        allocators.at(memloc.loc).free(memloc.offset, del_id);

        task_tensor_to_mem_node_erase_on_memory(task_id);

        return true;
    }
    return false;
}

void memgraph_make_state_t::task_tensor_to_mem_node_insert_on_storage(int tid, int mid)
{
    _task_tensor_to_mem_node_insert(tid, mid);
    tensors_on_storage.insert(tid);
}

void memgraph_make_state_t::task_tensor_to_mem_node_insert_on_memory(int tid, int mid)
{
    _task_tensor_to_mem_node_insert(tid, mid);
    tensors_on_memory.insert({tid, {}});
}

void memgraph_make_state_t::_task_tensor_to_mem_node_insert(int tid, int mid)
{
    if (task_tensor_to_mem_node.count(tid) > 0) {
        throw std::runtime_error("this tid is already in task_tensor_to_mem_node");
    }
    task_tensor_to_mem_node.insert({tid, mid});
}

void memgraph_make_state_t::task_tensor_to_mem_node_update_on_storage(int tid, int mid)
{
    _task_tensor_to_mem_node_update(tid, mid);

    {
        auto iter = tensors_on_memory.find(tid);
        if (iter != tensors_on_memory.end()) {
            tensors_on_memory.erase(iter);
            tensors_on_storage.insert(tid);
        } else {
            if (tensors_on_storage.count(tid) == 0) {
                throw std::runtime_error("update_on_storage: not in tensors_on_x");
            }
        }
    }
}

void memgraph_make_state_t::task_tensor_to_mem_node_update_on_memory(int tid, int mid)
{
    _task_tensor_to_mem_node_update(tid, mid);

    {
        auto iter_s = tensors_on_storage.find(tid);
        if (iter_s != tensors_on_storage.end()) {
            tensors_on_storage.erase(iter_s);
            tensors_on_memory.insert({tid, {}});
        } else {
            auto iter_m = tensors_on_memory.find(tid);
            if (iter_m == tensors_on_memory.end()) {
                throw std::runtime_error("update_on_memory: not in tensors_on_x");
            }
        }
    }
}

void memgraph_make_state_t::_task_tensor_to_mem_node_update(int tid, int mid)
{
    auto iter = task_tensor_to_mem_node.find(tid);
    if (iter == task_tensor_to_mem_node.end()) {
        throw std::runtime_error("when updating: tid is not in task_tensor_to_mem_node");
    }
    iter->second = mid;
}

void memgraph_make_state_t::task_tensor_to_mem_node_erase_on_storage(int tid)
{
    _task_tensor_to_mem_node_erase(tid);

    auto iter = tensors_on_storage.find(tid);
    if (iter == tensors_on_storage.end()) {
        throw std::runtime_error("cannot erase: not on tensors_on_storage");
    }
    tensors_on_storage.erase(iter);
}

void memgraph_make_state_t::task_tensor_to_mem_node_erase_on_memory(int tid)
{
    _task_tensor_to_mem_node_erase(tid);

    auto iter = tensors_on_memory.find(tid);
    if (iter == tensors_on_memory.end()) {
        throw std::runtime_error("cannot erase: not on tensors_on_memory");
    }
    tensors_on_memory.erase(iter);
}

void memgraph_make_state_t::_task_tensor_to_mem_node_erase(int tid)
{
    auto iter = task_tensor_to_mem_node.find(tid);
    if (iter == task_tensor_to_mem_node.end()) {
        throw std::runtime_error("cannot erase: not on task_tensor_to_mem_ndoe");
    }
    task_tensor_to_mem_node.erase(iter);
}

int memgraph_make_state_t::order_state_t::get(int tid)
{
    auto& xs = when_used.at(tid);
    auto  iter = xs.begin();
    while (iter != xs.end()) {
        if (*iter < threshold) {
            iter = xs.erase(iter);
        } else {
            return *iter;
        }
    }

    // If this tensor will not be used again but hasn't been deleted,
    // it is probably a save node.

    // Return a large value to say this "will never be used again"
    return std::numeric_limits<int>::max();
}

optional<vector<int>>
memgraph_make_state_t::find_victim(int loc, uint64_t size, vector<int> cannot_evict)
{
    // std::cout << "cannot evict: " << cannot_evict << std::endl;
    // form a bidirectional mapping block_id <-> tid, use _get_block_id
    map<int, int> bid2tid; // block id to tid
    map<int, int> tid2bid; // tid to block id
    for (auto const& [tid, mid] : task_tensor_to_mem_node) {
        memstoloc_t memstoloc = memgraph.nodes[mid].op.get_output_memstoloc();
        if (memstoloc.is_memloc()) {
            auto const& memloc = memstoloc.get_memloc();
            if (memloc.loc == loc) {
                int block_id = allocators.at(loc)._get_block_id(memloc.offset);
                bid2tid[block_id] = tid;
                tid2bid[tid] = block_id;
            }
        } else {
            // it is on storage, so we won't be evicting it
        }
    }

    // Get the set of block ids to evict from
    // _find_best_evict_block_ids.
    // Use order_state in the overload function.
    auto f_score = [&, this, cannot_evict](int bid) {
        // std::cout << "aaaa new inside lambda cannot evict: " << cannot_evict << std::endl;
        int  tid = bid2tid.at(bid);
        auto iter = std::find(cannot_evict.begin(), cannot_evict.end(), tid);
        if (iter != cannot_evict.end()) {
            return -1;
        }
        return order_state.value().get(tid);
    };

    auto maybe_evict_block_ids = allocators.at(loc)._find_best_evict_block_ids(size, f_score);

    if (!maybe_evict_block_ids) {
        // this should be unlikely

        return std::nullopt;
    }

    // using the block ids to evict, get the corresponding tids
    auto const& evict_block_ids = maybe_evict_block_ids.value();
    vector<int> evict_tids;
    evict_tids.reserve(evict_block_ids.size());
    for (auto bid : evict_block_ids) {
        evict_tids.push_back(bid2tid.at(bid));
    }

    // check that all the victims are not things we can't evict
    for (auto tid : evict_tids) {
        auto find_iter = std::find(cannot_evict.begin(), cannot_evict.end(), tid);
        if (find_iter != cannot_evict.end()) {
            throw std::runtime_error("thing we are evicting is in cannot evict!");
        }
    }

    return evict_tids;
}
// optional<int>
// memgraph_make_state_t::find_victim(
//   int loc,
//   uint64_t size,
//   vector<int> cannot_evict)
// {
//   if(!use_storage)
//   {
//     return std::nullopt;
//   }
//   if(!order_state)
//   {
//     throw std::runtime_error("order state must be setup");
//   }

//   auto& ostate = order_state.value();

//   vector<int> candidates = map_get_keys(tensors_on_memory);

//   vector_filter_inplace(candidates, [&](int const& tid)
//   {
//     // If this tid is in cannot_evict, return false
//     auto iter = std::find(cannot_evict.begin(), cannot_evict.end(), tid);
//     if(iter != cannot_evict.end()) {
//       return false;
//     }

//     if(taskgraph.nodes[tid].op.out_loc() != loc) {
//       return false;
//     }

//     // Make sure it is big enough to evict
//     uint64_t tensor_size = taskgraph.nodes[tid].op.out_size();
//     return tensor_size >= size;
//   });

//   // Order the candidates in ascending order so that the last item
//   // is the one that is used the lastest
//   vector_sort_inplace(candidates,
//     [&, this](int const& lhs, int const& rhs) {
//       return ostate.get(lhs) < ostate.get(rhs);
//     }
//   );

//   if(candidates.size() == 0)
//   {
//     return std::nullopt;
//   }

//   return candidates.back();
// }

int memgraph_make_state_t::allocate_with_evict(int loc, uint64_t size, vector<int> cannot_evict)
{
    {
        auto maybe = allocate_without_evict(loc, size);
        if (maybe) {
            return maybe.value();
        }
    }

    if (!use_storage) {
        throw std::runtime_error(
            "allocate_with_evict: would have to use storage but storage is unavailable.");
    }

    auto maybe_victims = find_victim(loc, size, cannot_evict);
    if (maybe_victims) {
        vector<int> const& victims = maybe_victims.value();
        for (auto vic_tid : victims) {
            evict_tensor(vic_tid);
        }
        auto maybe_ret = allocate_without_evict(loc, size);
        if (!maybe_ret) {
            throw std::runtime_error(
                "allocate_with_evict: could not allocate even after evicting a tensor"
                "... Is this an alignment issue?");
        }
        return maybe_ret.value();
    } else {
        throw std::runtime_error(
            "allocate_with_evict: could not allocate even after evicting a tensor.");
    }
}

optional<int> memgraph_make_state_t::allocate_without_evict(int loc, uint64_t size)
{
    auto maybe = allocators.at(loc).allocate(size);
    if (maybe) {
        auto const& [offset, deps] = maybe.value();
        alloc_t alloc{.loc = loc, .offset = offset, .size = size};
        int     new_memid = memgraph.insert(op_t(alloc), deps);
        return new_memid;
    } else {
        return std::nullopt;
    }
}

void memgraph_make_state_t::evict_tensor(int victim_tid)
{
    int         node_mid = task_tensor_to_mem_node.at(victim_tid);
    auto const& node = memgraph.nodes.at(node_mid);

    auto       evict_memloc = node.op.get_output_memloc();
    int const& storage_loc = memgraph.storage_locs.at(evict_memloc.loc);
    evict_t    evict{.src = evict_memloc, .dst = stoloc_t{.loc = storage_loc, .id = _sto_id}};
    _sto_id += 1;

    // When can an eviction begin? When all of the nodes that have used
    // victim_tid have been completed.
    //
    // Consider
    //   input[T] -> A -> B -> C -> D[use T]
    //            -> X -> Y -> Z -> ^
    // The outs of input[T] are just A and X but it is used
    // at D. So we want
    //   input[T] -> A -> B -> C -> D[use T] -> evict[T]
    //            -> X -> Y -> Z -> ^
    // NOT
    //   input[T] -> A -> B -> C -> D[use T]
    //               |> evict[T]
    //            -> X -> Y -> Z -> ^
    //   (the |> arrow means A to evict[T] and X to evict[T])
    //
    // This is why we go through the hassle of maintaining, for
    // all tensors in memory, where they've been used while in memory.
    //
    // Note that victim_tid may not actually have been used, in particular
    // it might be a save node that we are evicting because it won't be
    // used again.
    set<int>& evict_deps = tensors_on_memory.at(victim_tid);
    evict_deps.insert(node_mid); // in case it wasn't used

    int evict_mid = memgraph.insert(evict, evict_deps);

    // now free the memory, depending on the eviction having been completed
    allocators.at(evict_memloc.loc).free(evict_memloc.offset, evict_mid);

    // update mapping in memgraph_make_state_t
    task_tensor_to_mem_node_update_on_storage(victim_tid, evict_mid);
}

void memgraph_make_state_t::load_tensor_with_evict(int tid, vector<int> cannot_evict)
{
    if (tensors_on_storage.count(tid) == 0) {
        throw std::runtime_error("cannot load: not on storage");
    }

    auto const& node = taskgraph.nodes[tid];
    int         loc = node.op.out_loc();
    uint64_t    size = node.op.out_size();

    int alloc_mid = allocate_with_evict(loc, size, cannot_evict);
    _load_tensor_helper(tid, alloc_mid);
}

bool memgraph_make_state_t::load_tensor_without_evict(int tid)
{
    if (tensors_on_storage.count(tid) == 0) {
        throw std::runtime_error("cannot load: not on storage");
    }

    auto const& node = taskgraph.nodes[tid];
    int         loc = node.op.out_loc();
    uint64_t    size = node.op.out_size();

    auto maybe_alloc_mid = allocate_without_evict(loc, size);
    if (maybe_alloc_mid) {
        int const& alloc_mid = maybe_alloc_mid.value();
        _load_tensor_helper(tid, alloc_mid);
        return true;
    } else {
        return false;
    }
}

void memgraph_make_state_t::_load_tensor_helper(int tid, int alloc_mid)
{
    int const& sto_mid = task_tensor_to_mem_node.at(tid);

    load_t load{.src = memgraph.nodes.at(sto_mid).op.get_stoloc(),
                .dst = memgraph.nodes.at(alloc_mid).op.get_output_memloc()};

    int mid = memgraph.insert(op_t(load), set<int>{alloc_mid, sto_mid});
    task_tensor_to_mem_node_update_on_memory(tid, mid);
}

vector<tuple<int, mem_t>>
memgraph_make_state_t::get_tensors_in_memory_without_alloc(vector<int> const& task_ids)
{
    vector<tuple<int, mem_t>> ret;
    for (auto const& tid : task_ids) {
        auto iter = task_tensor_to_mem_node.find(tid);
        if (iter != task_tensor_to_mem_node.end()) {
            int const& memid = iter->second;
            auto       maybe_mem = memgraph.nodes[memid].op.get_output_memstoloc();
            if (maybe_mem.is_memloc()) {
                ret.emplace_back(memid, maybe_mem.get_memloc().as_mem());
            } else {
                throw std::runtime_error(
                    "get_tensors_in_memory_without_alloc: "
                    "tid not in memory, tid is on storage.");
            }
        } else {
            throw std::runtime_error("get_tensors_in_memory_without_alloc: tid not exist yet. ");
        }
    }

    return ret;
}

bool operator==(_which_node_t const& lhs, _which_node_t const& rhs)
{
    return lhs.task_id == rhs.task_id;
}
bool operator<(_which_node_t const& lhs, _which_node_t const& rhs)
{
    return lhs.task_id < rhs.task_id;
}

bool operator==(_which_touch_t const& lhs, _which_touch_t const& rhs)
{
    return three_tuple_eq(lhs, rhs);
}
bool operator<(_which_touch_t const& lhs, _which_touch_t const& rhs)
{
    return three_tuple_lt(lhs, rhs);
}