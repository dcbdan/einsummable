#include "graph.h"
#include "../base/hrect.h"

int graph_constructor_t::insert_input(placement_t placement, dtype_t dtype)
{
    int ret = graph.insert_input(placement.total_shape(), dtype);
    placements.insert({ret, placement});
    return ret;
}

int graph_constructor_t::insert_input(partition_t partition, dtype_t dtype)
{
    return insert_input(placement_t(partition), dtype);
}

int graph_constructor_t::insert_input(vector<uint64_t> shape, dtype_t dtype)
{
    return insert_input(partition_t::singleton(shape), dtype);
}

int graph_t::insert_input(vector<uint64_t> shape, dtype_t dtype)
{
    return this->insert(input_t{.dtype = dtype, .shape = shape}, {});
}

int graph_constructor_t::insert_einsummable(placement_t   placement,
                                            einsummable_t e,
                                            vector<int>   inns)
{
    if (placement.total_shape() != e.join_shape) {
        throw std::runtime_error("graph constructor: invalid insert_einsummable inputs");
    }

    int ret = graph.insert_einsummable(e, inns);
    placements.insert({ret, placement});
    return ret;
}

int graph_constructor_t::insert_einsummable(partition_t   partition,
                                            einsummable_t e,
                                            vector<int>   inns)
{
    return insert_einsummable(placement_t(partition), e, inns);
}

int graph_constructor_t::insert_einsummable(einsummable_t e, vector<int> inns)
{
    auto const& shape = e.join_shape;
    return insert_einsummable(partition_t::singleton(shape), e, inns);
}

int graph_t::insert_einsummable(einsummable_t e, vector<int> inns)
{
    if (e.inns.size() != inns.size()) {
        throw std::runtime_error("did not get expected number of inputs");
    }

    auto expected_inn_shapes = e.inn_shapes();
    auto expected_inn_dtypes = e.inn_dtypes();

    for (int i = 0; i != inns.size(); ++i) {
        if (!vector_equal(expected_inn_shapes[i], out_shape(inns[i]))) {
            throw std::runtime_error("shapes do not match: insert einsummable");
        }
        if (expected_inn_dtypes[i] != out_dtype(inns[i])) {
            throw std::runtime_error("dtype error in graph insert einsumable");
        }
    }

    return this->insert(e, inns);
}

int graph_constructor_t::insert_formation(placement_t placement, int inn, bool is_save)
{
    if (!vector_equal(placement.total_shape(), graph.out_shape(inn))) {
        throw std::runtime_error("invalid shape: insert_formation (constructing)");
    }

    int ret = graph.insert_formation(inn, is_save);
    placements.insert({ret, placement});
    return ret;
}

int graph_constructor_t::insert_formation(partition_t partition, int inn, bool is_save)
{
    return this->insert_formation(placement_t(partition), inn, is_save);
}

int graph_constructor_t::insert_formation(int inn, bool is_save)
{
    auto const& inn_node = graph.nodes[inn];
    auto        shape = inn_node.op.out_shape();
    return this->insert_formation(partition_t::singleton(shape), inn, is_save);
}

int graph_t::insert_formation(int inn, bool is_save)
{
    return this->insert(
        op_t(formation_t{.dtype = out_dtype(inn), .shape = out_shape(inn)}, is_save), {inn});
}

int graph_constructor_t::insert_to_complex(placement_t placement, int inn)
{
    int ret = graph.insert_to_complex(inn);
    if (!vector_equal(graph.out_shape(ret), placement.total_shape())) {
        throw std::runtime_error("invalid shape: insert_to_complex (constructing)");
    }

    placements.insert({ret, placement});
    return ret;
}

int graph_constructor_t::insert_to_complex(partition_t partition, int inn)
{
    return this->insert_to_complex(placement_t(partition), inn);
}

int graph_constructor_t::insert_to_complex(int inn)
{
    auto shape = graph.out_shape(inn);
    shape.back() /= 2;
    return this->insert_to_complex(partition_t::singleton(shape), inn);
}

int graph_t::insert_to_complex(int inn)
{
    if (out_dtype(inn) != dtype_t::f32) {
        throw std::runtime_error("can only convert to dtype_t::c64");
    }
    vector<uint64_t> shape = out_shape(inn);
    if (shape.back() % 2 == 1) {
        throw std::runtime_error("must have last even last input dim");
    }
    shape.back() /= 2;

    return this->insert(complexer_t{.dtype = dtype_t::c64, .shape = shape}, {inn});
}

int graph_t::insert_squeezer(vector<uint64_t> const& out_shape, int inn)
{
    squeezer_t squeezer{
        .dtype = this->out_dtype(inn), .inn_shape = this->out_shape(inn), .out_shape = out_shape};

    // TODO: This restriction may be too strict. Complex squeezers are not allowed
    //       because it complicates partitioning rules with respect to the last dimension.
    if (dtype_is_complex(squeezer.dtype)) {
        throw std::runtime_error(
            "graph_t::insert_squeezer: do not apply squeezer to complex valued tensors");
    }

    auto get_core = [](vector<uint64_t> const& ds) {
        vector<uint64_t> ret;
        ret.reserve(ds.size());
        for (auto const& d : ds) {
            if (d != 1) {
                ret.push_back(d);
            }
        }
        return ret;
    };

    if (!vector_equal(get_core(squeezer.inn_shape), get_core(squeezer.out_shape))) {
        throw std::runtime_error("squeezer given invalid reshaping " +
                                 write_with_ss(squeezer.inn_shape) + "->" +
                                 write_with_ss(squeezer.out_shape));
    }

    return this->insert(squeezer, {inn});
}

int graph_constructor_t::insert_to_real(placement_t placement, int inn)
{
    int ret = graph.insert_to_real(inn);
    if (!vector_equal(graph.out_shape(ret), placement.total_shape())) {
        throw std::runtime_error("invalid shape: insert_to_real (constructing)");
    }

    placements.insert({ret, placement});
    return ret;
}

int graph_constructor_t::insert_to_real(partition_t partition, int inn)
{
    return this->insert_to_real(placement_t(partition), inn);
}

int graph_constructor_t::insert_to_real(int inn)
{
    auto shape = graph.out_shape(inn);
    shape.back() *= 2;
    return this->insert_to_real(partition_t::singleton(shape), inn);
}

int graph_t::insert_to_real(int inn)
{
    if (out_dtype(inn) != dtype_t::c64) {
        throw std::runtime_error("can only convert from dtype_t::c64");
    }
    vector<uint64_t> shape = out_shape(inn);
    shape.back() *= 2;

    return this->insert(complexer_t{.dtype = dtype_t::f32, .shape = shape}, {inn});
}

int graph_t::insert_constant(scalar_t value, vector<uint64_t> const& shape)
{
    return insert_fill(constant_t{.value = value, .shape = shape});
}

int graph_t::insert_fill(fill_t const& fill)
{
    auto shape = fill.shape();
    if (shape.size() == 0) {
        throw std::runtime_error("invalid fill");
    }
    for (auto const& dim : shape) {
        if (dim == 0) {
            throw std::runtime_error("invalid dim in fill");
        }
    }

    return this->insert(fill, {});
}

int graph_constructor_t::insert_concat(placement_t placement, int dim, vector<int> inns)
{
    int ret = graph.insert_concat(dim, inns);

    if (placement.total_shape() != graph.out_shape(ret)) {
        throw std::runtime_error("graph constructor: invalid concat");
    }

    placements.insert({ret, placement});
    return ret;
}

int graph_constructor_t::insert_concat(partition_t partition, int dim, vector<int> inns)
{
    int ret = graph.insert_concat(dim, inns);

    if (partition.total_shape() != graph.out_shape(ret)) {
        throw std::runtime_error("graph constructor: invalid concat");
    }

    placements.insert({ret, placement_t(partition)});
    return ret;
}

int graph_constructor_t::insert_concat(int dim, vector<int> inns)
{
    int ret = graph.insert_concat(dim, inns);

    partition_t partition = partition_t::singleton(graph.out_shape(ret));

    placements.insert({ret, placement_t(partition)});
    return ret;
}

int graph_t::insert_concat(int dim, vector<int> inns)
{
    if (inns.size() <= 1) {
        throw std::runtime_error("concat must have multiple arguments");
    }

    dtype_t dtype = out_dtype(inns[0]);
    for (int i = 1; i != inns.size(); ++i) {
        if (out_dtype(inns[i]) != dtype) {
            throw std::runtime_error("dtype error at insert_concat");
        }
    }

    vector<vector<uint64_t>> shapes;
    for (int const& inn : inns) {
        shapes.push_back(out_shape(inn));
    }

    return this->insert(make_concat(dim, dtype, shapes), inns);
}

int graph_constructor_t::insert_subset(placement_t                       placement,
                                       vector<tuple<uint64_t, uint64_t>> hrect,
                                       int                               inn)
{
    int ret = graph.insert_subset(hrect, inn);

    if (placement.total_shape() != graph.out_shape(ret)) {
        throw std::runtime_error("graph constructor: invalid subset");
    }

    placements.insert({ret, placement});
    return ret;
}

int graph_constructor_t::insert_subset(partition_t                       partition,
                                       vector<tuple<uint64_t, uint64_t>> hrect,
                                       int                               inn)
{
    int ret = graph.insert_subset(hrect, inn);

    if (partition.total_shape() != graph.out_shape(ret)) {
        throw std::runtime_error("graph constructor: invalid subset");
    }

    placements.insert({ret, placement_t(partition)});
    return ret;
}

int graph_constructor_t::insert_subset(vector<tuple<uint64_t, uint64_t>> hrect, int inn)
{
    int ret = graph.insert_subset(hrect, inn);

    partition_t partition = partition_t::singleton(graph.out_shape(ret));

    placements.insert({ret, placement_t(partition)});
    return ret;
}

int graph_t::insert_subset(vector<tuple<uint64_t, uint64_t>> hrect, int inn)
{
    dtype_t          dtype = out_dtype(inn);
    vector<uint64_t> inn_shape = out_shape(inn);

    return this->insert(make_subset(dtype, hrect, inn_shape), {inn});
}

dtype_t graph_t::complexer_t::inn_dtype() const
{
    if (dtype == dtype_t::f32) {
        return dtype_t::c64;
    }
    if (dtype == dtype_t::c64) {
        return dtype_t::f32;
    }
    throw std::runtime_error("inn_dtype complexer: invalid dtype");
}

vector<uint64_t> graph_t::complexer_t::inn_shape() const
{
    vector<uint64_t> ret = shape;
    if (dtype_is_real(dtype)) {
        ret.back() *= 2;
        return ret;
    } else if (dtype_is_complex(dtype)) {
        if (ret.back() % 2 == 1) {
            throw std::runtime_error("invalid complexer shape");
        }
        ret.back() /= 2;
        return ret;
    } else {
        throw std::runtime_error("should not reach");
    }
}

graph_t::op_t::op_t(graph_t::op_t::_op_t op_, bool s) : op(op_), is_save_(s)
{
    if (has_aggregation() && is_save()) {
        throw std::runtime_error("an einsummable with an aggregation cannot be saved");
    }
}

dtype_t graph_t::op_t::out_dtype() const
{
    if (is_input()) {
        return get_input().dtype;
    }
    if (is_formation()) {
        return get_formation().dtype;
    }
    if (is_complexer()) {
        return get_complexer().dtype;
    }
    if (is_squeezer()) {
        return get_squeezer().dtype;
    }
    if (is_fill()) {
        return get_fill().dtype();
    }
    if (is_select()) {
        return get_select().dtype;
    }
    if (is_einsummable()) {
        return get_einsummable().out_dtype();
    }
    throw std::runtime_error("graph::op_t should not reach");
}

void graph_t::op_t::set_save(bool s)
{
    if (has_aggregation() && s) {
        throw std::runtime_error(
            "set_save: "
            "an einsummable with an aggregation cannot be saved");
    }
    is_save_ = s;
}

vector<uint64_t> graph_t::op_t::out_shape() const
{
    if (is_input()) {
        return get_input().shape;
    }
    if (is_formation()) {
        return get_formation().shape;
    }
    if (is_complexer()) {
        return get_complexer().shape;
    }
    if (is_squeezer()) {
        return get_squeezer().out_shape;
    }
    if (is_fill()) {
        return get_fill().shape();
    }
    if (is_select()) {
        return get_select().out_shape;
    }
    if (is_einsummable()) {
        return get_einsummable().out_shape();
    }
    throw std::runtime_error("graph::op_t should not reach");
}

vector<uint64_t> graph_t::op_t::shape() const
{
    if (is_input()) {
        return get_input().shape;
    }
    if (is_formation()) {
        return get_formation().shape;
    }
    if (is_complexer()) {
        return get_complexer().shape;
    }
    if (is_squeezer()) {
        return get_squeezer().out_shape;
    }
    if (is_fill()) {
        return get_fill().shape();
    }
    if (is_select()) {
        return get_select().out_shape;
    }
    if (is_einsummable()) {
        return get_einsummable().join_shape;
    }
    throw std::runtime_error("graph::op_t should not reach");
}

vector<placement_t> graph_constructor_t::get_placements() const
{
    vector<placement_t> ret;
    ret.reserve(graph.nodes.size());
    for (int gid = 0; gid != graph.nodes.size(); ++gid) {
        ret.push_back(placements.at(gid));
    }
    return ret;
}

void graph_t::set_saves()
{
    int num_nodes_time_zero = nodes.size();
    for (int i = 0; i != num_nodes_time_zero; ++i) {
        node_t& n = nodes[i];
        if (n.outs.size() == 0 && !n.op.is_save()) {
            if (n.op.has_aggregation()) {
                this->insert_formation(i, true);
            } else {
                n.op.set_save(true);
            }
        }
    }
}

vector<partition_t> graph_t::make_singleton_partition() const
{
    vector<partition_t> ps;
    ps.reserve(nodes.size());
    for (int gid = 0; gid != nodes.size(); ++gid) {
        auto const& node = nodes[gid];
        ps.push_back(partition_t::singleton(node.op.shape()));
    }
    return ps;
}

vector<placement_t> graph_t::make_singleton_placement() const
{
    vector<placement_t> pls;
    pls.reserve(nodes.size());
    for (auto const& part : make_singleton_partition()) {
        pls.emplace_back(part);
    }
    return pls;
}

vector<uint64_t> graph_t::out_shape(int id) const
{
    return nodes[id].op.out_shape();
}

dtype_t graph_t::out_dtype(int id) const
{
    return nodes[id].op.out_dtype();
}

vector<int> graph_t::get_order() const
{
    // Because of the way the graph is constructed,
    // it must be the case that a valid ordering of the compute
    // graph is 0,1,2,...
    vector<int> ret(nodes.size());
    std::iota(ret.begin(), ret.end(), 0);
    return ret;
}

vector<int> graph_t::get_flop_order() const
{
    vector<tuple<uint64_t, int>> pending;
    vector<int>                  cnts;
    for (int gid = 0; gid != nodes.size(); ++gid) {
        int num_deps = nodes[gid].get_inns_set().size();
        if (num_deps == 0) {
            // Assuming that any node with no inputs has no flops
            pending.emplace_back(0, gid);
        }
        cnts.push_back(num_deps);
    }

    auto pop = [&] {
        tuple<uint64_t, int> ret = pending.back();
        pending.pop_back();
        return ret;
    };

    auto insert = [&](uint64_t f, int id) {
        tuple<uint64_t, int> item(f, id);
        pending.insert(std::lower_bound(pending.begin(),
                                        pending.end(),
                                        item,
                                        [](tuple<uint64_t, int> const& x,
                                           tuple<uint64_t, int> const& y) { return x > y; }),
                       item);
    };

    auto decrement = [&](uint64_t v) {
        for (auto& [rem, _] : pending) {
            rem -= v;
        }
    };

    auto get_flops = [&](int id) -> uint64_t {
        auto const& node = nodes[id];
        if (node.op.is_einsummable()) {
            return product(node.op.get_einsummable().join_shape);
        }
        return 0;
    };

    auto complete = [&](int id) {
        auto const& node = nodes[id];
        for (int out : node.outs) {
            int& cnt = cnts[out];
            cnt -= 1;
            if (cnt == 0) {
                insert(get_flops(out), out);
            }
        }
    };

    vector<int> ret;
    while (pending.size() > 0) {
        auto [rem, id] = pop();
        ret.push_back(id);
        decrement(rem);
        complete(id);
    }

    if (ret.size() != nodes.size()) {
        throw std::runtime_error("not all nodes included: get_flop_order");
    }

    return ret;
}

vector<int> graph_t::get_reverse_order() const
{
    // Can't tell if this is just the reverse of get_order() or not,
    // so wrote the full algorithm
    vector<int> ret;
    // reserve to not invalidate iterators
    ret.reserve(nodes.size());

    vector<int> deps;
    deps.reserve(nodes.size());
    for (int gid = 0; gid != nodes.size(); ++gid) {
        auto const& node = nodes[gid];
        int         ndep = node.outs.size();
        if (ndep == 0) {
            ret.push_back(gid);
        }
        deps[gid] = ndep;
    }

    for (auto iter = ret.begin(); iter != ret.end(); ++iter) {
        int         gid = *iter;
        auto const& node = nodes[gid];
        set<int>    inns(node.inns.begin(), node.inns.end());
        for (auto const& out_gid : inns) {
            int& cnt = deps[out_gid];
            cnt--;
            if (cnt == 0) {
                ret.push_back(out_gid);
            }
        }
    }

    return ret;
}

int graph_t::insert(op_t const& op, vector<int> inns)
{
    int ret = nodes.size();
    nodes.push_back(node_t{.op = op, .inns = inns, .outs = {}});

    for (auto inn : inns) {
        nodes.at(inn).outs.insert(ret);
    }

    return ret;
}

void graph_t::print() const
{
    std::cout << "graph[num nodes = " << nodes.size() << "]" << std::endl;
    std::cout << std::endl;

    for (int id = 0; id != nodes.size(); ++id) {
        auto const& node = nodes[id];

        std::cout << "node id: " << id << " with out shape " << node.op.out_shape() << " | "
                  << node.op.out_dtype() << std::endl;
        std::cout << "inputs: " << node.inns << std::endl;
        if (node.op.is_input()) {
            std::cout << "input" << std::endl;
        } else if (node.op.is_einsummable()) {
            std::cout << "einsummable " << node.op.get_einsummable() << std::endl;
        } else if (node.op.is_formation()) {
            std::cout << "formation (is save = " << std::boolalpha << node.op.is_save() << ")"
                      << std::endl;
        } else if (node.op.is_complexer()) {
            if (node.op.get_complexer().is_to_real()) {
                std::cout << "complexer (to real)" << std::endl;
            } else {
                std::cout << "complexer (to complex)" << std::endl;
            }
        } else if (node.op.is_squeezer()) {
            std::cout << "squeezer" << std::endl;
        } else if (node.op.is_fill()) {
            auto const& f = node.op.get_fill();
            string      s;
            if (f.is_constant()) {
                s = write_with_ss(f.get_constant().value);
            } else if (f.is_lowertri()) {
                s = "lowertri";
            } else {
                throw std::runtime_error("missing fill type");
            }
            std::cout << "fill[" << s << "]" << std::endl;
        } else if (node.op.is_select()) {
            std::cout << "select" << std::endl;
        } else {
            throw std::runtime_error("graph_t print should not reach");
        }

        std::cout << std::endl;
    }
}

void graph_t::print_graphviz(std::ostream& out, map<int, string> get_color) const
{
    print_graphviz(out, make_singleton_partition(), get_color);
}

void graph_t::print_graphviz(std::ostream&              out,
                             vector<partition_t> const& parts,
                             map<int, string>           get_color) const
{
    using std::endl;
    string tab = "  ";
    out << "digraph {" << endl;

    for (int id = 0; id != nodes.size(); ++id) {
        node_t const& node = nodes[id];
        op_t const&   op = node.op;

        string label;
        string color = "";
        if (op.is_input()) {
            label = "input" + write_with_ss(id);
            label += "\n" + write_with_ss(op.shape());
            color = "green";
        } else if (op.is_formation()) {
            label = "form" + write_with_ss(id);
            label += "\n" + write_with_ss(op.out_shape());
        } else if (op.is_complexer()) {
            label = "complexer" + write_with_ss(id);
            color = "purple";
        } else if (op.is_squeezer()) {
            label = "squeezer" + write_with_ss(id);
            color = "azure2";
        } else if (op.is_einsummable()) {
            auto const& e = op.get_einsummable();
            label = "einsummable" + write_with_ss(id) + ":" + e.str();
            if (e.is_contraction()) {
                color = "pink";
            }
            label += "\n" + e.join.to_cppstr() + "  |  " + write_with_ss(e.castable);
        } else if (op.is_fill()) {
            auto const& f = node.op.get_fill();
            label = write_with_ss(f);
            color = "brown";
        } else if (op.is_select()) {
            label = "select" + write_with_ss(id);
            color = "gold";
        } else {
            throw std::runtime_error("printgraphviz missing graph node type");
        }
        label += "\n" + write_with_ss(parts[id].block_shape());
        // label += "\n" + write_with_ss(parts[id].total_shape()) + ":" +
        // write_with_ss(out_dtype(id));
        out << tab << "n" << id << " [style=filled,label=\"" << label << "\"";

        // set the color with get_color as precedent
        {
            auto iter = get_color.find(id);
            if (iter != get_color.end()) {
                color = iter->second;
            }
        }

        if (color != "") {
            out << ",color=\"" << color << "\"";
        }
        out << "]" << endl;

        int _i = 0;
        for (auto const& inn : node.inns) {
            out << tab << "n" << inn << " -> " << "n" << id << "[label=\"" << write_with_ss(_i++)
                << "\"]" << endl;
        }
    }
    out << "}" << endl;
}

void graph_t::print_subset_graphviz(std::ostream&              out,
                                    set<int> const&            gids,
                                    vector<partition_t> const& parts) const
{
    using std::endl;
    string tab = "  ";
    out << "digraph {" << endl;

    set<int> extras;
    for (int id = 0; id != nodes.size(); ++id) {
        if (gids.count(id) == 0) {
            continue;
        }

        node_t const& node = nodes[id];
        op_t const&   op = node.op;

        string label;
        string color = "";
        if (op.is_input()) {
            label = "input" + write_with_ss(id);
            label += "\n" + write_with_ss(op.shape());
            color = "green";
        } else if (op.is_formation()) {
            label = "form" + write_with_ss(id);
            label += "\n" + write_with_ss(op.out_shape());
        } else if (op.is_complexer()) {
            label = "complexer" + write_with_ss(id);
        } else if (op.is_squeezer()) {
            label = "squeezer" + write_with_ss(id);
            color = "azure2";
        } else if (op.is_einsummable()) {
            auto const& e = op.get_einsummable();
            label = "einsummable" + write_with_ss(id) + ":" + e.str();
            if (e.is_contraction()) {
                color = "pink";
            }
            label += "\n" + e.join.to_cppstr() + "  |  " + write_with_ss(e.castable);
        } else if (op.is_fill()) {
            auto const& f = node.op.get_fill();
            label = write_with_ss(f);
        } else if (op.is_select()) {
            label = "select" + write_with_ss(id);
        } else {
            throw std::runtime_error("printgraphviz missing graph node type");
        }
        label += "\n" + write_with_ss(parts[id].block_shape());
        out << tab << "n" << id << " [style=filled,label=\"" << label << "\"";

        if (color != "") {
            out << ",color=\"" << color << "\"";
        }
        out << "]" << endl;

        int _i = 0;
        for (auto const& inn : node.inns) {
            if (gids.count(inn) == 0 && extras.count(inn) == 0) {
                extras.insert(inn);

                string _label = "";
                string _color = "";

                _label = "subset_input" + write_with_ss(inn);
                _label += "\n" + write_with_ss(op.out_shape());
                _color = "silver";
                _label += "\n" + write_with_ss(parts[inn].block_shape());

                out << tab << "n" << inn << " [style=filled,label=\"" << _label << "\"";
                if (_color != "") {
                    out << ",color=\"" << _color << "\"";
                }
                out << "]" << endl;
            }

            out << tab << "n" << inn << " -> " << "n" << id << "[label=\"" << write_with_ss(_i++)
                << "\"]" << endl;
        }
    }

    out << "}" << endl;
}

vector<int> graph_t::get_inputs() const
{
    vector<int> ret;
    for (int id = 0; id != nodes.size(); ++id) {
        auto const& node = nodes[id];
        if (node.op.is_input()) {
            ret.push_back(id);
        }
    }
    return ret;
}

// Construct a 3D matmul graph, (ij,jk->ik)
//   shape lhs: di*pi x dj*pj
//   shape rhs: dj*pj x dk*pk
//   shape out: di*pi x dk*pk
graph_constructor_t three_dimensional_matrix_multiplication(int      pi,
                                                            int      pj,
                                                            int      pk,
                                                            uint64_t di,
                                                            uint64_t dj,
                                                            uint64_t dk,
                                                            int      num_processors)
{
    // The mapping from "procesor" grid to actual processor;
    // this is necessary for when pi*pj*pk > num_processors
    auto to_processor = [&](int i, int j, int k) {
        int index = idxs_to_index({pi, pj, pk}, {i, j, k});
        return index % num_processors;
    };

    // rcp = row, column, row_part
    enum class rcp_t { ijk, jki, ikj };

    // All matrices are partitioned along the rows and then the
    // columns, but then each block is further partitioned.
    //
    // So if A is partitioned rcp_t::ijk, then there are pi rows,
    // pj columns to form Aij. But each Aij is partitioned further
    // along the rows, into pk parts.
    // That means the partition is really (pi*pk, pj).
    auto make_matrix_partition = [&](rcp_t which) {
        int      nr;
        int      nc;
        int      np;
        uint64_t dr;
        uint64_t dc;
        if (which == rcp_t::ijk) {
            nr = pi;
            nc = pj;
            np = pk;
            dr = di;
            dc = dj;
        } else if (which == rcp_t::jki) {
            nr = pj;
            nc = pk;
            np = pi;
            dr = dj;
            dc = dk;
        } else if (which == rcp_t::ikj) {
            nr = pi;
            nc = pk;
            np = pj;
            dr = di;
            dc = dk;
        } else {
            throw std::runtime_error("should not reach");
        }

        // Take dr, repeat it nr times,
        // then each of those nr blocks gets
        // split np ways.
        partdim_t part_row = partdim_t::split_each(partdim_t::repeat(nr, dr), np);

        partdim_t part_col = partdim_t::repeat(nc, dc);

        return partition_t({part_row, part_col});
    };

    // For which == rcp_t::ijk,
    // Aij(k) lives at processor (i,j,k).
    // That is, A is partitioned into pi rows, pj columns.
    // Each Aij is distributed across (i,j,*) and
    // Aij is chopped along it's rows forming Aij(k) for
    // some k in 0,...,pk-1.
    auto make_matrix_locs = [&](rcp_t which) {
        vector<int> shape;
        int         nr;
        int         nc;
        int         np;
        if (which == rcp_t::ijk) {
            shape = {pi * pk, pj};
            nr = pi;
            nc = pj;
            np = pk;
        } else if (which == rcp_t::jki) {
            shape = {pj * pi, pk};
            nr = pj;
            nc = pk;
            np = pi;
        } else if (which == rcp_t::ikj) {
            shape = {pi * pj, pk};
            nr = pi;
            nc = pk;
            np = pj;
        } else {
            throw std::runtime_error("should not reach");
        }
        vtensor_t<int> locs(shape);

        int i;
        int j;
        int k;
        for (int r = 0; r != nr; ++r) {
            for (int c = 0; c != nc; ++c) {
                for (int p = 0; p != np; ++p) {
                    if (which == rcp_t::ijk) {
                        i = r;
                        j = c;
                        k = p;
                    } else if (which == rcp_t::jki) {
                        i = p;
                        j = r;
                        k = c;
                    } else if (which == rcp_t::ikj) {
                        i = r;
                        j = p;
                        k = c;
                    } else {
                        throw std::runtime_error("should not reach");
                    }

                    locs(r * np + p, c) = to_processor(i, j, k);
                }
            }
        }

        return locs;
    };

    auto make_matrix_placement = [&](rcp_t rcp) {
        return placement_t(make_matrix_partition(rcp), make_matrix_locs(rcp));
    };

    graph_constructor_t ret;

    int id_lhs = ret.insert_input(make_matrix_placement(rcp_t::ijk));
    int id_rhs = ret.insert_input(make_matrix_placement(rcp_t::jki));

    int id_op;
    {
        einsummable_t matmul = einsummable_t::from_matmul(di * pi, dj * pj, dk * pk);
        // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

        partition_t part(
            {partdim_t::repeat(pi, di), partdim_t::repeat(pk, dk), partdim_t::repeat(pj, dj)});

        vtensor_t<int> locs({pi, pk, pj});

        for (int i = 0; i != pi; ++i) {
            for (int j = 0; j != pj; ++j) {
                for (int k = 0; k != pk; ++k) {
                    locs(i, k, j) = to_processor(i, j, k);
                }
            }
        }

        placement_t placement(part, locs);

        id_op = ret.insert_einsummable(placement, matmul, {id_lhs, id_rhs});
    }

    // the save node
    ret.insert_formation(make_matrix_placement(rcp_t::ikj), id_op, true);

    return ret;
}

graph_constructor_t
straight_matrix_multiplication(int pi, int pj, int pk, uint64_t di, uint64_t dj, uint64_t dk)
{
    graph_constructor_t graph;

    partdim_t pdi = partdim_t::repeat(pi, di);
    partdim_t pdj = partdim_t::repeat(pj, dj);
    partdim_t pdk = partdim_t::repeat(pk, dk);

    int id_lhs = graph.insert_input(partition_t({pdi, pdj}));
    int id_rhs = graph.insert_input(partition_t({pdj, pdk}));

    einsummable_t matmul = einsummable_t::from_matmul(pi * di, pj * dj, pk * dk);
    // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

    int id_join = graph.insert_einsummable(partition_t({pdi, pdk, pdj}), matmul, {id_lhs, id_rhs});

    graph.insert_formation(partition_t({pdi, pdk}), id_join, true);

    return graph;
}

tuple<vector<tuple<int, int>>, graph_constructor_t>
create_remap_graph_constructor(remap_relations_t const& _remap)
{
    auto const& remap = _remap.remap;

    graph_constructor_t g;

    vector<tuple<int, int>> remap_gid;
    for (auto const& [src, dst] : remap) {
        int gid_src = g.insert_input(src.placement, src.dtype);
        int gid_dst = g.insert_formation(dst.placement, gid_src, true);
        remap_gid.emplace_back(gid_src, gid_dst);
    }

    return {remap_gid, g};
}
