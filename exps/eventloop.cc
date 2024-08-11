#include "../src/engine/exec_state.h"

void main01()
{
    exec_graph_t g;

    using op_ptr_t = exec_graph_t::op_ptr_t;
    using dummy_t = exec_graph_t::dummy_t;

    for (int i = 0; i != 10; ++i) {
        op_ptr_t op = std::make_shared<dummy_t>();
        g.insert(op, {});
        g.nodes[i].priority = i;
    }

    rm_ptr_t resource_manager = rm_ptr_t(new resource_manager_t({}));

    {
        exec_state_t state(g, resource_manager);
        state.event_loop();
    }
}

struct silly_resource_t : rm_template_t<unit_t, unit_t> {
    silly_resource_t(int c = 1) : count(c) {}

    static desc_ptr_t make_desc()
    {
        return rm_template_t::make_desc(unit_t{});
    }

private:
    int count;

    optional<unit_t> try_to_acquire_impl(unit_t const&)
    {
        if (count == 0) {
            return std::nullopt;
        }

        count--;

        return unit_t{};
    }

    void release_impl(unit_t const&)
    {
        count++;
    }
};

struct silly_op_t : exec_graph_t::op_base_t {
    silly_op_t(int p = 99) : priority(p) {}

    void launch(resource_ptr_t resource, std::function<void()> callback) const
    {
        callback();
    }

    desc_ptr_t resource_description() const
    {
        return silly_resource_t::make_desc();
    }

    void print(std::ostream& out) const
    {
        out << "silly_op_t";
    }

    int get_priority() const
    {
        return priority;
    }

    int priority;
};

void main02()
{
    exec_graph_t g;

    using op_ptr_t = exec_graph_t::op_ptr_t;

    g.insert(std::make_shared<silly_op_t>(), {});     // 0
    g.insert(std::make_shared<silly_op_t>(), {});     // 1
    g.insert(std::make_shared<silly_op_t>(), {0, 1}); // 2

    g.insert(std::make_shared<silly_op_t>(), {});     // 3
    g.insert(std::make_shared<silly_op_t>(), {});     // 4
    g.insert(std::make_shared<silly_op_t>(), {3, 4}); // 5

    g.insert(std::make_shared<silly_op_t>(), {2, 5}); // 6

    rm_ptr_t resource_manager = rm_ptr_t(new silly_resource_t());

    DOUT("GIVEN");
    {
        exec_state_t state(g, resource_manager, exec_state_t::priority_t::given);
        state.event_loop();
    }

    DOUT("BFS");
    {
        exec_state_t state(g, resource_manager, exec_state_t::priority_t::bfs);
        // possible execution : 0,1,3,4,2,5,6
        //                      0,4,3,1,2,5,6
        state.event_loop();
    }

    DOUT("DFS");
    {
        exec_state_t state(g, resource_manager, exec_state_t::priority_t::dfs);
        // possible execution : 0,1,2,3,4,5,6
        //                      4,0,1,3,5,2,6
        state.event_loop();
    }
}

int main()
{
    main02();
}
