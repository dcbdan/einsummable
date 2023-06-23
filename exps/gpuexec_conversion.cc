#include "../src/einsummable/einsummable.h"

#include "../src/execution/gpu/kernels.h"

#include "../src/einsummable/reference.h"
#include "../src/einsummable/scalarop.h"

#include <chrono>

#include <unordered_map>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

void print_cutensor_op(cutensorOperator_t op){

    if(op==CUTENSOR_OP_ADD){
        printf("CUTENSOR_OP_ADD\n");
    }else if(op==CUTENSOR_OP_MUL){
        printf("CUTENSOR_OP_MUL\n");
    }else if(op==CUTENSOR_OP_EXP){
        printf("CUTENSOR_OP_EXP\n");
    }else if(op==CUTENSOR_OP_IDENTITY){
        printf("CUTENSOR_OP_IDENTITY\n");
    }


}



int main(){
    //einsummable_t e = einsummable_t({96,96,96,64,64,64}, { {0,4,5,2},{1,5,3,4} }, 4, scalarop_t::make_mul(dtype_t::f32), castable_t::add);
    einsummable_t e = einsummable_t({96,96,96,64,64,64}, { {0,4,5,2},{1,5,3,4},{0,1,2,3} }, 4, scalarop_t::combine(scalarop_t::make_mul(), { scalarop_t::make_add(), scalarop_t::make_exp() }), castable_t::add);
     
    auto ceot = make_cutensor_elementwise_op(e);    

    printf("scalarop_t::combine(scalarop_t::make_mul(), { scalarop_t::make_add(), scalarop_t::make_exp() })\n");

    cutensor_elementwise_op_t op = *ceot;

    if(std::holds_alternative<cutensor_elementwise_op_t::unary_t>(op.op)){
        printf("This is unary\n");
        auto unary = std::get<cutensor_elementwise_op_t::unary_t>(op.op);
    }else if(std::holds_alternative<cutensor_elementwise_op_t::binary_t>(op.op)){
        printf("This is binary\n");
        printf("OP is\n");
        auto binary = std::get<cutensor_elementwise_op_t::binary_t>(op.op);
        print_cutensor_op(binary.op);
        printf("Arg 1\n");
        printf("Scale is: %.2f\n",binary.lhs.scale.f32());
        printf("OP is ");
        print_cutensor_op(binary.lhs.op);
        printf("Arg 2\n");
        printf("Scale is: %.2f\n",binary.rhs.scale.f32());
        printf("OP is ");
        print_cutensor_op(binary.rhs.op);
    }else if(std::holds_alternative<cutensor_elementwise_op_t::ternary_t>(op.op)){
        printf("This is ternary\n");
        printf("OP_01_2 is\n");
        auto ternary = std::get<cutensor_elementwise_op_t::ternary_t>(op.op);
        print_cutensor_op(ternary.op_01_2);
        printf("OP_0_1 is\n");
        print_cutensor_op(ternary.op_0_1);
        printf("Arg 1\n");
        printf("Scale is: %.2f\n",ternary.a0.scale.f32());
        printf("OP is ");
        print_cutensor_op(ternary.a0.op);
        printf("Arg 2\n");
        printf("Scale is: %.2f\n",ternary.a1.scale.f32());
        printf("OP is ");
        print_cutensor_op(ternary.a1.op);
        printf("Arg 3\n");
        printf("Scale is: %.2f\n",ternary.a2.scale.f32());
        printf("OP is ");
        print_cutensor_op(ternary.a2.op);
        

    }
}