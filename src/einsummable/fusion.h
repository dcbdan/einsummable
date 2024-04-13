#pragma once

#include "graph.h"

class fusion_t {
public:
    graph_t apply(const graph_t& graph);

private:
    static bool is_exp(const einsummable_t& e);
    static bool is_relu(const einsummable_t& e);
    std::vector<std::pair<int, int>> findFusibleNodes(const graph_t& graph);
    einsummable_t createFusedEinsummable(const einsummable_t& non_elementwise, const einsummable_t& elementwise);
    graph_t fuseNodes(const graph_t& graph, const std::vector<std::pair<int, int>>& fusiblePairs);
    
};
