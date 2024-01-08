#include "modules.h"
#include "builder.h"

#include "../src/einsummable/reference.h"

#include <fstream>

void print_torch_tensor(string name, dbuffer_t d, vector<uint64_t> const& shape) {
  auto& out = std::cout;
  float const* v = d.f32();
  out << name << " = torch.tensor([" << v[0];
  for(int i = 1; i != d.nelem(); ++i) {
    out << "," << v[i];
  }
  out << "], dtype = torch.float32).reshape([";
  out << shape[0];
  for(int i = 1; i != shape.size(); ++i) {
    out << "," << shape[i];
  }
  out << "])";
  out << std::endl;
}

void insert_data(map<int, dbuffer_t>& data, graph_t const& graph, int id) {
  dtype_t dtype = graph.out_dtype(id);
  uint64_t nelem = product(graph.out_shape(id));
  dbuffer_t ret = make_dbuffer(dtype, nelem);
  ret.random("-0.1", "0.1");
  auto [_, did_insert] = data.insert({id, ret});
  if(!did_insert) {
    throw std::runtime_error("already have this id!");
  }
}

void rms_norm_test() {
  full_dim_t dim = full_dim_t::singleton(5);
  full_shape_t shape({dim, dim, dim});
  graph_writer_t writer;
  rms_norm_t norm(
    &writer,
    "rms_norm",
    dim,
    1e-6);
  auto x = writer.input(shape);
  auto y = norm.forward(x);

  auto grad = writer.backprop(y, {norm.weight})[0];
  grad.save_inplace();

  auto const& graph = writer.get_graph();

  map<int, dbuffer_t> data;
  insert_data(data, graph, x.get_id());
  insert_data(data, graph, norm.weight.get_id());

  auto p = [&](string name, int id) {
    print_torch_tensor(name, data.at(id), graph.out_shape(id));
  };
  p("x", x.get_id());
  p("weight", norm.weight.get_id());

  data = reference_compute_graph(graph, data);
  p("grad", grad.get_id());
}

dbuffer_t make_dbuffer_from_vector(vector<float> const& ds) {
  dbuffer_t ret = make_dbuffer(dtype_t::f32, ds.size());
  for(int i = 0; i != ds.size(); ++i) {
    ret.f32()[i] = ds[i];
  }
  return ret;
}
dbuffer_t make_dbuffer_from_vector(vector<double> const& ds) {
  dbuffer_t ret = make_dbuffer(dtype_t::f64, ds.size());
  for(int i = 0; i != ds.size(); ++i) {
    ret.f64()[i] = ds[i];
  }
  return ret;
}

void attention_test() {
vector<float> wq_{0.2499602735042572,-0.13315677642822266,-0.10071790218353271,0.126775860786438,-0.11707538366317749,-0.10703837126493454,-0.2570668160915375,-0.1562623828649521,0.30586540699005127,0.020009595900774002,-0.30571722984313965,-0.18106268346309662,0.025621827691793442,0.17702017724514008,0.04787856712937355,-0.20070599019527435,-0.21316172182559967,-0.08537261188030243,0.32395070791244507,0.22086651623249054,-0.2821596562862396,0.11415136605501175,-0.0019673726055771112,-0.12169425934553146,0.0740460678935051,0.2565458416938782,0.08549399673938751,-0.11749693751335144,-0.1747431457042694,-0.01299336738884449,-0.23192015290260315,0.35020506381988525,-0.23338349163532257,0.19346916675567627,0.2073984444141388,0.057634931057691574,0.18139703571796417,-0.006720525678247213,0.31468459963798523,0.1588968187570572,-0.3474109470844269,-0.0968591496348381,-0.18571649491786957,0.2206977903842926,0.06551697850227356,0.3151071071624756,-0.28831300139427185,-0.19362059235572815,-0.3189006745815277,0.3028104305267334,0.0024692995939403772,0.06799833476543427,-0.2004750669002533,0.1800188422203064,-0.24311178922653198,0.1272958666086197,-0.023037297651171684,0.10995829850435257,-0.01935787871479988,0.27207502722740173,0.1594601571559906,-0.17916464805603027,-0.24846616387367249,0.001341871335171163};
vector<float> wk_{0.040540166199207306,0.2627137303352356,0.13104769587516785,-0.26626715064048767,-0.08550798892974854,0.20748965442180634,0.31241849064826965,0.07190631330013275,-0.07386530190706253,0.27191680669784546,-0.3302239179611206,0.346051961183548,0.290885329246521,-0.023068908601999283,-0.17065384984016418,0.3035837411880493,0.22352218627929688,0.20760728418827057,-0.07496246695518494,-0.2619287371635437,-0.04072599485516548,-0.18141566216945648,0.14678950607776642,-0.17481441795825958,0.17395058274269104,-0.0483747199177742,-0.09890921413898468,-0.26692166924476624,-0.3481133282184601,-0.017806200310587883,0.21231146156787872,0.2872389554977417,0.2767120599746704,-0.09577028453350067,0.11454042047262192,0.0005083752912469208,-0.1432507336139679,0.010276919230818748,-0.11418811976909637,0.20283663272857666,0.30185699462890625,0.2657982409000397,0.12149401754140854,0.3004663586616516,0.0850059762597084,-0.008802833035588264,-0.2194761335849762,-0.07834348827600479,0.3385298550128937,0.013499129563570023,-0.14500103890895844,0.30858901143074036,0.02114992029964924,0.07648266106843948,-0.19368137419223785,-0.3304711878299713,0.2531939744949341,-0.16884158551692963,0.08349555730819702,0.1539822816848755,0.15126903355121613,-0.3191169202327728,-0.17436623573303223,0.13155405223369598};
vector<float> wv_{0.22776582837104797,0.0811297744512558,0.26371827721595764,-0.17846639454364777,0.12184742093086243,-0.13074441254138947,-0.0003111701807938516,0.26529037952423096,0.026959525421261787,-0.2930002212524414,0.06394102424383163,-0.31172117590904236,-0.14719508588314056,0.13069729506969452,0.0645885244011879,-0.2752566933631897,0.2768666744232178,0.09824033826589584,-0.18707047402858734,-0.05613867565989494,-0.11327285319566727,-0.27741509675979614,0.3456026315689087,0.19541950523853302,0.3500964045524597,-0.006709356792271137,0.2685299515724182,0.2092982977628708,0.270134836435318,0.045287124812603,-0.3494201600551605,0.3101929724216461,0.2018204778432846,-0.0684157982468605,-0.2731448709964752,0.10017083585262299,-0.02490389719605446,-0.2363092452287674,0.0817614272236824,-0.18352115154266357,-0.13140885531902313,0.25440385937690735,-0.22633345425128937,-0.19026941061019897,0.27778294682502747,-0.22584767639636993,0.28733184933662415,0.02838556468486786,0.3161441683769226,-0.30727314949035645,-0.20324984192848206,0.29377907514572144,0.2677467465400696,-0.3492740988731384,0.05510219931602478,-0.05996813625097275,0.1510685533285141,-0.03673553094267845,0.29615768790245056,0.29730015993118286,-0.1895153671503067,-0.2775772511959076,0.2927209138870239,0.20029239356517792};
vector<float> wo_{-0.2306288629770279,-0.051286980509757996,-0.26567989587783813,-0.01384991779923439,0.20028401911258698,-0.02188871242105961,-0.16600054502487183,0.033333394676446915,-0.01703689433634281,-0.28968003392219543,0.016840700060129166,-0.02840377204120159,0.10658280551433563,0.10051049292087555,0.3158060312271118,0.16460628807544708,-0.1205332800745964,-0.2002246230840683,-0.03768792375922203,0.032190412282943726,0.28317007422447205,0.17259037494659424,-0.26660671830177307,0.27032551169395447,0.19019679725170135,-0.3315296471118927,-0.11548665910959244,0.1143389642238617,-0.14806099236011505,0.21907328069210052,0.19158697128295898,0.2153136134147644,-0.18247178196907043,-0.16335001587867737,0.0283507090061903,-0.04379061609506607,0.10310476273298264,-0.10505451261997223,-0.2088662087917328,-0.08974143117666245,-0.03296937048435211,0.05237019807100296,-0.17869555950164795,0.14704234898090363,-0.22887840867042542,0.1719520092010498,0.05899277329444885,-0.15945088863372803,0.12014847993850708,0.13399232923984528,0.234992116689682,-0.3230211138725281,0.04800652340054512,0.11729269474744797,-0.23999229073524475,-0.19043083488941193,-0.20028865337371826,0.25957489013671875,0.20613841712474823,0.34745582938194275,-0.0428730808198452,-0.156558096408844,0.17168189585208893,0.19712510704994202};
vector<float> x_{0.3381996154785156,0.9548467993736267,0.9769788384437561,0.3396027088165283,0.18365108966827393,0.16593164205551147,0.9781875610351562,0.007507979869842529,0.23603081703186035,0.9373998641967773,0.42554235458374023,0.6075173020362854,0.21578317880630493,0.16626644134521484,0.10066843032836914,0.9671133160591125,0.9549607038497925,0.018408656120300293,0.7054814696311951,0.493960976600647,0.3584657311439514,0.8800656795501709,0.8932123184204102,0.7592791318893433};

  set_default_dtype(dtype_t::f32);

  dbuffer_t _wq = make_dbuffer_from_vector(wq_);
  dbuffer_t _wk = make_dbuffer_from_vector(wk_);
  dbuffer_t _wv = make_dbuffer_from_vector(wv_);
  dbuffer_t _wo = make_dbuffer_from_vector(wo_);
  dbuffer_t _x  = make_dbuffer_from_vector(x_);

  model_args_t args = model_args_t::llama(1, 1);
  args.dim = 8;
  args.n_heads = 2;
  args.max_seq_len = 3;

  graph_writer_t writer;
  attention_t attention(&writer, "attention", args, 0);
  auto x = writer.input({ {1},{3}, {2,4} }, default_dtype());
  dbuffer_t freqs_cis_data = transformer_t::form_freqs_cis(4,3);
  auto freqs_cis = writer.input({3,2}, dtype_t::c64);
  auto y = attention.forward(x, freqs_cis, std::nullopt);

  map<int, dbuffer_t> data;
  DOUT("x has shape " << x.get_shape());
  data.insert({x.get_id(), _x});
  data.insert({freqs_cis.get_id(), freqs_cis_data});
  data.insert({attention.wq.get_id(), _wq});
  data.insert({attention.wk.get_id(), _wk});
  data.insert({attention.wv.get_id(), _wv});
  data.insert({attention.wo.get_id(), _wo});

  auto gs = writer.backprop(y, {attention.wq, attention.wk, attention.wv, attention.wo});
  for(auto& g: gs) {
    g.save_inplace();
  }

  {
    map<int, string> colors;
    std::ofstream f("g.gv");
    writer.get_graph().print_graphviz(f, colors);
    DOUT("printed g.gv");
  }

  data = reference_compute_graph(writer.get_graph(), data);
  vector<string> names{"wq", "wk", "wv", "wo"};
  DOUT("freqs cis is " << freqs_cis_data);
  for(int i = 0; i != 4; ++i) {
    auto const& g = gs[i];
    auto const& str = names[i];
    int id = g.get_id();
    DOUT(id << " | " << str << ": " << data.at(id).sum_to_f64());
  }
}

void softmax_test() {
  set_default_dtype(dtype_t::f64);
  vector<double> x_{0.9,0.3,-0.3,0.1};
  vector<double> y_{0.1,0.2,0.3,0.4};

  dbuffer_t _x = make_dbuffer_from_vector(x_);
  dbuffer_t _y = make_dbuffer_from_vector(y_);

  uint64_t d = 2;
  graph_writer_t writer;
  auto x = writer.input({d,d});
  auto y = writer.input(vector<uint64_t>{d,d});
  auto z = writer.ew(
    "ij,ij->ij",
    scalarop_t::make_mul(default_dtype()),
    writer.softmax(x), y);
  z.save_inplace();

  map<int, string> colors;
  for(int i = 0; i != writer.get_graph().nodes.size(); ++i) {
    colors.insert({i, "azure"});
  }

  auto grads = writer.backprop(z, {x,y});
  auto gx = grads[0];
  auto gy = grads[1];
  gx.save_inplace();
  gy.save_inplace();

  auto const& graph = writer.get_graph();

  {
    std::ofstream f("g.gv");
    writer.get_graph().print_graphviz(f, colors);
    DOUT("printed g.gv");
  }

  map<int, dbuffer_t> data;
  data.insert({x.get_id(), _x});
  data.insert({y.get_id(), _y});

  data = reference_compute_graph(graph, data);

  int gx_id = gx.get_id();
  int gy_id = gy.get_id();
  DOUT("gx " << data.at(gx_id));
  DOUT("gy " << data.at(gy_id));
}

void reduction_test() {
vector<float> x_{10.032285690307617,10.384787559509277,10.385295867919922,10.182760238647461,10.585400581359863,10.280753135681152,10.527616500854492,10.478981018066406,10.055124282836914,10.519526481628418,10.70089340209961,10.455738067626953,10.353894233703613,10.742764472961426,10.403480529785156,10.907011032104492,10.126445770263672,10.058435440063477,10.92620849609375,10.344030380249023,10.58475112915039,10.767372131347656,10.464858055114746,10.78731918334961,10.23349380493164};
vector<float> y_{10.98783016204834,10.890084266662598,10.28034496307373,10.278607368469238,10.670238494873047,10.82606029510498,10.129566192626953,10.238995552062988,10.635088920593262,10.774934768676758,10.389111518859863,10.083562850952148,10.276042938232422,10.155156135559082,10.998201370239258,10.164838790893555,10.766170501708984,10.01076889038086,10.9969482421875,10.388680458068848,10.330077171325684,10.250794410705566,10.466246604919434,10.995975494384766,10.698726654052734};


  dbuffer_t _x = make_dbuffer_from_vector(x_);
  dbuffer_t _y = make_dbuffer_from_vector(y_);

  uint64_t d = 5;
  graph_writer_t writer;
  auto x = writer.input({d,d});
  auto y = writer.input({d,d});
  auto z = writer.matmul(x,y);
  //z = writer.reduction("ij->i", castable_t::add, z);
  z = writer.reduction("ij->i", castable_t::max, z);
  z.save_inplace();
  auto grads = writer.backprop(z, {x,y});
  auto gx = grads[0];
  auto gy = grads[1];
  gx.save_inplace();
  gy.save_inplace();

  auto const& graph = writer.get_graph();
  {
    map<int, string> colors;
    std::ofstream f("g.gv");
    writer.get_graph().print_graphviz(f, colors);
    DOUT("printed g.gv");
  }

  map<int, dbuffer_t> data;
  data.insert({x.get_id(), _x});
  data.insert({y.get_id(), _y});

  data = reference_compute_graph(graph, data);

  int gx_id = gx.get_id();
  int gy_id = gy.get_id();
  DOUT(data.at(z.get_id()));
  DOUT(gx_id << " gx " << data.at(gx_id).sum());
  DOUT(gy_id << " gy " << data.at(gy_id).sum());

}

void reduction_test2() {
  uint64_t d = 3;
  graph_writer_t writer;
  auto x = writer.input({d,d});
  auto z = writer.reduction("ij->i", castable_t::max, x);
  z.save_inplace();
  auto gx = writer.backprop(z, {x})[0];
  gx.save_inplace();

  auto const& graph = writer.get_graph();
  {
    map<int, string> colors;
    std::ofstream f("g.gv");
    writer.get_graph().print_graphviz(f, colors);
    DOUT("printed g.gv");
  }

  map<int, dbuffer_t> data;

  dbuffer_t ones = make_dbuffer(default_dtype(), d*d);
  ones.ones();
  DOUT(ones);
  data.insert({x.get_id(), ones});

  data = reference_compute_graph(graph, data);

  DOUT(data.at(z.get_id()));
  DOUT(data.at(gx.get_id()));
}

void complex_test() {
  vector<float> x_{0.10576850175857544,0.49335700273513794};
  vector<float> y_{5.0,3.0};//{0.8755072951316833,0.7137557864189148};

  DOUT("y: " << y_);

  dbuffer_t _x  = make_dbuffer_from_vector(x_);
  dbuffer_t _y  = make_dbuffer_from_vector(y_).view_f32_as_c64();

  graph_writer_t writer;
  auto x = writer.input({1,2});
  auto y = writer.input({1,1}, dtype_t::c64);

  // Case 1:
  DOUT("(x.to_complex() * y).to_real()");
  auto z = writer.mul(x.to_complex(), y).to_real();
  // correct: (y_r+y_c, y_r-y_c)

  // Case 2:
  // DOUT("(x.to_complex() * z).to_real()");
  //auto z = writer.mul(x.to_complex(), y);
  //correct: grad x = y.to_real()

  map<int, dbuffer_t> data;
  data.insert({x.get_id(), _x});
  data.insert({y.get_id(), _y});

  auto gx = writer.backprop(z, {x})[0];
  gx.save_inplace();

  {
    map<int, string> colors;
    std::ofstream f("g.gv");
    writer.get_graph().print_graphviz(f, colors);
    DOUT("printed g.gv");
  }

  data = reference_compute_graph(writer.get_graph(), data);
  DOUT("grad x: " << data.at(gx.get_id()));
}

void complex_test2() {
  vector<float> x_{0.10576850175857544,0.49335700273513794};
  dbuffer_t _x  = make_dbuffer_from_vector(x_);

  graph_writer_t writer;
  auto x = writer.input({1,2});
  auto z = x.to_complex();

  map<int, dbuffer_t> data;
  data.insert({x.get_id(), _x});

  auto gx = writer.backprop(z, {x})[0];
  gx.save_inplace();

  {
    map<int, string> colors;
    std::ofstream f("g.gv");
    writer.get_graph().print_graphviz(f, colors);
    DOUT("printed g.gv");
  }

  data = reference_compute_graph(writer.get_graph(), data);
  DOUT(data.at(gx.get_id()));
  // 1+i0
}
void complex_test3() {
  vector<float> x_{0.10576850175857544,0.49335700273513794};
  dbuffer_t _x  = make_dbuffer_from_vector(x_).view_f32_as_c64();

  graph_writer_t writer;
  auto x = writer.input({1,1}, dtype_t::c64);
  auto z = x.to_real();

  DOUT("x.to_real()");
  map<int, dbuffer_t> data;
  data.insert({x.get_id(), _x});

  auto gx = writer.backprop(z, {x})[0];
  gx.save_inplace();

  {
    map<int, string> colors;
    std::ofstream f("g.gv");
    writer.get_graph().print_graphviz(f, colors);
    DOUT("printed g.gv");
  }

  data = reference_compute_graph(writer.get_graph(), data);
  DOUT("x:  " << x_);
  DOUT("gx: " << data.at(gx.get_id()));
  // 1 + i1
}

void softmax_test3() {
  set_default_dtype(dtype_t::f64);
  vector<double> x_{0.9,0.3,-0.3,0.1};
  vector<double> y_{0.1,0.2,0.3,0.4};

  dbuffer_t _x = make_dbuffer_from_vector(x_);
  dbuffer_t _y = make_dbuffer_from_vector(y_);

  uint64_t d = 2;
  graph_writer_t writer;
  auto x = writer.input({d,d});
  auto y = writer.input(vector<uint64_t>{d,d});
  auto z = writer.mul(
    writer.exp(x), y);
  z.save_inplace();

  map<int, string> colors;
  for(int i = 0; i != writer.get_graph().nodes.size(); ++i) {
    colors.insert({i, "azure"});
  }

  auto grads = writer.backprop(z, {x,y});
  auto gx = grads[0];
  auto gy = grads[1];
  gx.save_inplace();
  gy.save_inplace();

  auto const& graph = writer.get_graph();

  {
    std::ofstream f("g.gv");
    writer.get_graph().print_graphviz(f, colors);
    DOUT("printed g.gv");
  }

  map<int, dbuffer_t> data;
  data.insert({x.get_id(), _x});
  data.insert({y.get_id(), _y});

  data = reference_compute_graph(graph, data);

  int gx_id = gx.get_id();
  int gy_id = gy.get_id();

  DOUT("gx " << gx_id << ": " << data.at(gx_id));
  DOUT("gy " << gy_id << ": " << data.at(gy_id));
}

void forward_test1()
{
  uint64_t dn  = 4;
  uint64_t dp  = 2;
  uint64_t dd  = 2;
  uint64_t dw1 = 2;
  uint64_t dw2 = 2;

vector<float> x_{0.2633971571922302,0.2270956039428711,0.500491738319397,0.24498778581619263,0.6035093665122986,0.778448760509491,0.7928462028503418,0.05222243070602417};
vector<float> y_{0.36576616764068604,0.7166582345962524,0.5282002091407776,0.6736341118812561,0.1187816858291626,0.6779669523239136,0.1932758092880249,0.17560386657714844};
vector<float> w0_{0.05514711141586304,0.9867408871650696,0.37867271900177,0.45346295833587646};
vector<float> w1_{0.8223041892051697,0.3970258831977844,0.11585962772369385,0.5765251517295837};
vector<float> w2_{0.9273481965065002,0.6494401693344116,0.06650429964065552,0.23578673601150513};

  dbuffer_t _x = make_dbuffer_from_vector(x_);
  dbuffer_t _y = make_dbuffer_from_vector(y_);
  dbuffer_t _w0 = make_dbuffer_from_vector(w0_);
  dbuffer_t _w1 = make_dbuffer_from_vector(w1_);
  dbuffer_t _w2 = make_dbuffer_from_vector(w2_);

  graph_writer_t writer;
  auto x = writer.input({dn,dp});
  auto y = writer.input({dn,dd});
  auto w0 = writer.input({dp,dw1});
  auto w1 = writer.input({dw1,dw2});
  auto w2 = writer.input({dw2,dd});

  scalarop_t relu = scalarop_t::make_relu();

  auto z = writer.ew(relu, writer.matmul(x, w0));
  z = writer.ew(relu, writer.matmul(z, w1));
  z = writer.matmul(z, w2);

  string ds = write_with_ss(dtype_t::f32);
  scalarop_t squared_difference =
    scalarop_t::from_string(
      "power{2}[+[hole|"+ds+"@0,*[hole|"+ds+"@1,constant{"+ds+"|-1}]]]");
  auto loss = writer.ew("ij,ij->ij", squared_difference, z, y);
  loss = loss.scale(scalar_t(1/(float(1.0)*dn*dd)));

  map<int, string> colors;
  for(int i = 0; i != writer.get_graph().nodes.size(); ++i) {
    colors.insert({i, "azure"});
  }

  auto grads = writer.backprop(loss, {w0,w1,w2});
  auto gw0 = grads[0];
  auto gw1 = grads[1];
  auto gw2 = grads[2];
  gw0.save_inplace();
  gw1.save_inplace();
  gw2.save_inplace();

  auto const& graph = writer.get_graph();

  {
    std::ofstream f("g.gv");
    writer.get_graph().print_graphviz(f, colors);
    DOUT("printed g.gv");
  }

  map<int, dbuffer_t> data;
  data.insert({x.get_id(), _x});
  data.insert({y.get_id(), _y});
  data.insert({w0.get_id(), _w0});
  data.insert({w1.get_id(), _w1});
  data.insert({w2.get_id(), _w2});

  data = reference_compute_graph(graph, data);

  int gw0_id = gw0.get_id();
  int gw1_id = gw1.get_id();
  int gw2_id = gw2.get_id();

  DOUT("gw0 " << gw0_id << ": " << data.at(gw0_id));
  DOUT("gw1 " << gw1_id << ": " << data.at(gw1_id));
  DOUT("gw2 " << gw2_id << ": " << data.at(gw2_id));
}

int main() {
  set_default_dtype(dtype_t::f32);

  //rms_norm_test();
  //attention_test();

  //complex_test();
  //complex_test2();
  //DOUT("")
  //complex_test3();

  //softmax_test();
  //reduction_test();
  //reduction_test2();

  forward_test1();
}
