#include "ff.h"

ff_sqdiff_t
ff_sqdiff_update(
  uint64_t dn, uint64_t dp, uint64_t dd,
  vector<uint64_t> dws,
  float learning_rate)
{
  scalarop_t gradupdate = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::from_string("hole@0"),
      scalarop_t::make_scale(learning_rate)
    }
  );

  scalarop_t relu = scalarop_t::make_relu();

  scalarop_t squared_difference =
    scalarop_t::from_string("power{2}[+[hole@0,*[hole@1,constant{-1}]]]");

  matrixgraph_t mgraph;

  int x = mgraph.insert_input(dn, dp);
  int y = mgraph.insert_input(dn, dd);

  int yhat = x;
  vector<int> ws;
  vector<uint64_t> ws_sizes;
  {
    uint64_t dlast = dp;
    for(auto const& dw: dws) {
      ws.push_back(mgraph.insert_input(dlast, dw));
      ws_sizes.push_back(dlast*dw);
      yhat = mgraph.insert_matmul_ss(yhat, ws.back());
      yhat = mgraph.insert_ew(relu, yhat);
      dlast = dw;
    }
    ws.push_back(mgraph.insert_input(dlast, dd));
    ws_sizes.push_back(dlast*dd);
    yhat = mgraph.insert_matmul_ss(yhat, ws.back());
  }

  int sqdiff = mgraph.insert_ewb(squared_difference, yhat, y);

  vector<int> grads = mgraph.backprop(sqdiff, ws);

  vector<int> wsnew;
  for(int i = 0; i != ws.size(); ++i) {
    int const& g = grads[i];
    int const& w = ws[i];
    wsnew.push_back(mgraph.insert_ewb(gradupdate, w, g));
  }

  return ff_sqdiff_t {
    .mgraph = mgraph,
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

