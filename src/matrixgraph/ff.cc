#include "ff.h"

ff_sqdiff_t
ff_sqdiff_update(
  uint64_t dn, uint64_t dp, uint64_t dd,
  vector<uint64_t> dws,
  float learning_rate,
  dtype_t dtype)
{
  if(dtype == dtype_t::c64) {
    throw std::runtime_error("ff_sqdiff_update does not support complex");
  }

  dtype_t dtype_before = default_dtype();
  set_default_dtype(dtype);

  string ds = write_with_ss(dtype);

  scalar_t lr = scalar_t(learning_rate).convert(dtype);

  scalarop_t gradupdate = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::from_string("hole|"+ds+"@0"),
      scalarop_t::make_scale(lr)
    }
  );

  scalarop_t relu = scalarop_t::make_relu();

  scalarop_t squared_difference =
    scalarop_t::from_string(
      "power{2}[+[hole|"+ds+"@0,*[hole|"+ds+"@1,constant{"+ds+"|-1}]]]"); // (yhat - y)**2

  matrixgraph_t mgraph;

  int x = mgraph.insert_input(dn, dp);
  int y = mgraph.insert_input(dn, dd); 
  std::cout << "Inserted expected output matrix with dimensions: " << dn << ", " << dd << " " << y <<  std::endl;

  int yhat = x;
  vector<int> ws;
  vector<uint64_t> ws_sizes;
  {
    uint64_t dlast = dp;
    int i = 1;
    for(auto const& dw: dws) {
      ws.push_back(mgraph.insert_input(dlast, dw));
      std::cout << "Inserted weight matrix W" << i++ << " with dimensions: " << dlast << ", " << dw << std::endl;
      ws_sizes.push_back(dlast*dw);
      std::cout << "Inserting matmul node that has inn nodes: " << yhat << ", " << ws.back() << std::endl;
      yhat = mgraph.insert_matmul_ss(yhat, ws.back());
      std::cout << "Inserting ReLu node that has inn node: " << yhat << std::endl;
      yhat = mgraph.insert_ew(relu, yhat);
      
      dlast = dw;
    }
    ws.push_back(mgraph.insert_input(dlast, dd));
    std::cout << "Inserted weight matrix of outer layer W" << i++ << " with dimensions: " << dlast << ", " << dd << std::endl;
    ws_sizes.push_back(dlast*dd);
    std::cout << "Inserting matmul node that has inn nodes: " << yhat << ", " << ws.back() << std::endl;
    yhat = mgraph.insert_matmul_ss(yhat, ws.back());
  }

  std::cout << "Inserting squared difference node with inn nodes: " << yhat << ", " << y << std::endl; 
  int sqdiff = mgraph.insert_ewb(squared_difference, yhat, y);

  vector<int> grads = mgraph.backprop(sqdiff, ws);

  vector<int> wsnew;
  for(int i = 0; i != ws.size(); ++i) {
    int const& g = grads[i];
    int const& w = ws[i];
    wsnew.push_back(mgraph.insert_ewb(gradupdate, w, g));
    std::cout << "Inserted gradupdate node with entry nodes: " << w << ", " << g << std::endl; 
  }

  set_default_dtype(dtype_before);

  return ff_sqdiff_t {
    .mgraph = mgraph,
    .dtype = dtype,
    .dn = dn,
    .dp = dp,
    .dd = dd,
    .dws = dws,
    .learning_rate = learning_rate,
    .x = x,
    .y = y,
    .yhat = yhat,
    .sqdiff = sqdiff,
    .wsinn = ws,
    .wsout = wsnew,
    .grads = grads
  };
}

tuple<uint64_t,uint64_t> ff_sqdiff_t::shape_wi(int i) const {
  uint64_t d0,d1;
  if(i == 0) {
    d0 = dp;
  } else {
    d0 = dws[i-1];
  }

  if(i == dws.size()) {
    d1 = dd;
  } else {
    d1 = dws[i];
  }

  return {d0,d1};
}

