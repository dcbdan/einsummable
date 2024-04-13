#include "fusion.h"
#include "einsummable.h"
#include "graph.h"
#include <algorithm>


// Find pairs of nodes where an 'exp' node is followed by a 'ReLU' node.
std::vector<std::pair<int, int>> fusion_t::findFusibleNodes(const graph_t& graph) 
{
    std::vector<std::pair<int, int>> fusiblePairs;

    for (int i = 0; i < graph.nodes.size(); i++) {
        const auto& node = graph.nodes[i];

        if (node.op.is_einsummable()) 
        {
            const auto& einsum = node.op.get_einsummable();
            if (is_exp(einsum)) // Check for 'exp' operation
            {  
                for (int child_id : node.outs)  // Get the child nodes
                { 
                    const auto& child_node = graph.nodes[child_id];
                    if (child_node.op.is_einsummable() && is_relu(child_node.op.get_einsummable())) // Check for 'ReLU'
                    {  
                        fusiblePairs.emplace_back(i, child_id);
                    }
                }
            }
        }
    }

    return fusiblePairs;
}

graph_t fusion_t::apply(const graph_t& originalGraph) 
{
    auto fusiblePairs = findFusibleNodes(originalGraph);
    return fuseNodes(originalGraph, fusiblePairs);
}



graph_t fusion_t::fuseNodes(const graph_t& graph, const std::vector<std::pair<int, int>>& fusiblePairs) 
{
    // Start with a copy of the original graph
    graph_t newGraph = graph; 

    for (const auto& pair : fusiblePairs) 
    {
        auto& exp_node = newGraph.nodes[pair.first];
        auto& relu_node = newGraph.nodes[pair.second];

        // Create fused einsummable
        einsummable_t fused = createFusedEinsummable(exp_node.op.get_einsummable(), relu_node.op.get_einsummable());

        // Replace the 'exp' node with the fused operation in the new graph
        newGraph.nodes[pair.first].op = graph_t::op_t(fused);

        // Remove the 'ReLU' node from the new graph, and fix connections
        newGraph.removeNode(pair.second);
    }

    return newGraph;
}

// Create a fused einsummable that represents 'exp' followed by 'ReLU'.
einsummable_t fusion_t::createFusedEinsummable(const einsummable_t& exp, const einsummable_t& relu) 
{
    // Ensure the operations are 'exp' followed by 'ReLU'
    if (!is_exp(exp) || !is_relu(relu)) {
        throw std::runtime_error("Unsupported operations for fusion. Expected 'exp' followed by 'ReLU'.");
    }

    // Check if both operations are compatible for fusion
    if (exp.join_shape != relu.join_shape || exp.inns != relu.inns || exp.out_rank != relu.out_rank) {
        throw std::runtime_error("Exp is not compatable with ReLU");
    }

    // Create a new operation type that represents the fused 'exp' and 'ReLU'.
    scalarop_t fused_op = scalarop_t::make_exp_relu(); 

    // The fused einsummable will have the same input tensors as the 'exp' operation,
    // and it will use the newly created scalar operation.
    return einsummable_t(exp.join_shape, exp.inns, exp.out_rank, fused_op);
}

// TODO: Find a more Robust implementation
bool fusion_t::is_exp(const einsummable_t& e) {
    if (e.inns.size() == 1) {  // Assuming exp is a unary operation
        scalarop_t op = e.join;
        op = op.simplify();
        auto op_str = op.to_cppstr();
        // Check if the operation string starts with "_exp(x0)"
        if (op_str.find("_exp(x0)") == 0) { // TODO: This is NOT a good way. It can Fail easily
            return true;
        }
    }
    return false;
}

// TODO: Find a more Robust implementation
bool fusion_t::is_relu(const einsummable_t& e) {
    if (e.inns.size() == 1) {  // Assuming ReLU is a unary operation
        scalarop_t op = e.join;
        op = op.simplify();
        auto op_str = op.to_cppstr();
        // Check if the operation string represents ReLU, assuming ReLU is represented as "(x0 >= 0 ? x0 : 0)"
        if (op_str.find("(f32|0>=x0?f32|0:x0)") == 0) { // TODO: This is NOT a good way. It can Fail easily
            return true;
        }
    }
    return false;
}
