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
vector<float> wq_{0.01376768946647644,-0.12721161544322968,-0.02070813626050949,-0.17344506084918976,0.02197001315653324,-0.3145335912704468,0.26115190982818604,-0.3493688404560089,-0.08817373216152191,-0.20058536529541016,-0.10945325344800949,-0.14092805981636047,-0.008151452988386154,0.3167003393173218,0.2681388854980469,0.28125059604644775,0.009558062069118023,-0.2964266538619995,-0.1572095900774002,0.1797933131456375,-0.22496247291564941,0.2056216597557068,0.32806676626205444,-0.12227407097816467,-0.33306387066841125,0.2749798595905304,0.2954957187175751,-0.047985535115003586,0.31711217761039734,-0.32786324620246887,-0.17517481744289398,-0.24580062925815582,-0.29080283641815186,0.08705954253673553,-0.08052331954240799,0.29339128732681274,0.27478259801864624,-0.2879498302936554,0.03800613060593605,0.1252349317073822,0.24828265607357025,0.22365440428256989,-0.15900573134422302,-0.30042484402656555,0.03718135878443718,-0.074897900223732,-0.1701330840587616,0.06743554770946503,0.3124728500843048,0.07965724915266037,-0.22001610696315765,-0.27684539556503296,-0.12255313247442245,-0.10904692113399506,0.13012337684631348,0.13434576988220215,0.05561116337776184,0.24570462107658386,-0.14416906237602234,-0.1802438199520111,0.04610726237297058,0.21089759469032288,0.10740862786769867,-0.17827054858207703};
vector<float> wk_{0.008798491209745407,-0.19465921819210052,0.21998614072799683,-0.053052596747875214,-0.34836700558662415,0.2595669627189636,-0.056183602660894394,-0.15474291145801544,0.2614355981349945,-0.05866478756070137,0.022356795147061348,-0.12006448209285736,-0.3261987268924713,-0.006063961889594793,-0.1788201779127121,-0.017620375379920006,-0.04260460287332535,0.344491571187973,0.28541404008865356,-0.22080223262310028,-0.35018011927604675,0.2577500641345978,-0.1569199562072754,-0.17558389902114868,-0.1713496595621109,0.2809143364429474,-0.1643543243408203,-0.047582946717739105,-0.06750550866127014,-0.09107272326946259,0.03479791432619095,-0.0018017356051132083,-0.007829704321920872,0.05464579164981842,0.23032332956790924,0.29782965779304504,0.2785392105579376,0.2898072898387909,-0.22440111637115479,-0.22764916718006134,-0.033336006104946136,-0.009377758018672466,-0.010561452247202396,0.3370758295059204,-0.1789475530385971,0.11398029327392578,0.14666196703910828,-0.11685533821582794,-0.007637261878699064,-0.2526082396507263,-0.07724657654762268,-0.06634706258773804,0.1427990049123764,-0.10920100659132004,0.10806304216384888,0.2726648151874542,0.12248674780130386,0.34504806995391846,0.33021482825279236,0.14311914145946503,-0.304991751909256,-0.13772308826446533,0.31367748975753784,0.14195361733436584};
vector<float> wv_{0.23208734393119812,0.008127176202833652,0.3121567368507385,-0.015752552077174187,0.14172033965587616,0.0747886523604393,0.06954621523618698,0.08568201214075089,0.23641511797904968,0.07869954407215118,0.1272810399532318,0.06849857419729233,0.3043111264705658,0.05166512355208397,0.24939271807670593,0.3510858118534088,-0.1252957135438919,-0.08056062459945679,-0.011439075693488121,0.2914595603942871,0.15869957208633423,0.11358276009559631,-0.34018978476524353,0.279531329870224,-0.1485682725906372,0.33454474806785583,0.2528652846813202,-0.03976837545633316,-0.2703670263290405,0.19234620034694672,0.3488197922706604,0.03137917071580887,0.20657463371753693,-0.00041232260991819203,0.07848354429006577,-0.15182289481163025,-0.31928664445877075,-0.018547099083662033,-0.320766419172287,-0.016579432412981987,-0.042472388595342636,-0.14465923607349396,-0.047142598778009415,0.2275572419166565,-0.008296185173094273,-0.28752070665359497,-0.3452034294605255,0.262773722410202,-0.1389312744140625,0.06272340565919876,0.18734923005104065,0.1614443063735962,0.14187796413898468,0.3008379340171814,-0.03887123614549637,0.33834603428840637,-0.2080380618572235,-0.2628512680530548,0.3401605188846588,0.05210273340344429,0.03674349561333656,0.18615175783634186,0.22469820082187653,-0.3179472088813782};
vector<float> wo_{0.3007771372795105,0.22999092936515808,-0.1753469854593277,0.3533809185028076,-0.2515535354614258,-0.20351912081241608,-0.02997799962759018,0.3149491548538208,0.2223053127527237,0.11441625654697418,0.051549091935157776,0.25829410552978516,0.29542744159698486,-0.28745296597480774,-0.04886442422866821,0.2338123321533203,0.10056044161319733,-0.3097895681858063,0.23340970277786255,0.16897223889827728,0.12214168906211853,-0.24397331476211548,0.052212949842214584,-0.09755604714155197,0.09325926005840302,0.290667325258255,-0.3447696566581726,0.3324006497859955,0.25080496072769165,0.2360956370830536,0.09838385134935379,-0.06810471415519714,0.24899379909038544,-0.33824479579925537,0.14719828963279724,-0.014060525223612785,0.315496563911438,-0.14539912343025208,-0.1305708885192871,-0.24371035397052765,0.05280953645706177,0.07516764104366302,0.2456570714712143,-0.2598961889743805,0.2968903183937073,-0.2945317029953003,0.11157939583063126,-0.2494458705186844,-0.31110745668411255,-0.16253890097141266,0.15484502911567688,-0.3452475368976593,0.09512707591056824,0.13126331567764282,-0.2405463457107544,-0.0024062900338321924,-0.32433024048805237,-0.015112468972802162,-0.16224214434623718,-0.3140402138233185,0.15137121081352234,0.034571416676044464,0.09645520895719528,-0.034933626651763916};
vector<float> x_{0.767062783241272,0.6914229989051819,0.4184572100639343,0.4275180697441101,0.06776237487792969,0.5101518630981445,0.1493808627128601,0.8776070475578308,0.5686613321304321,0.7116783857345581,0.6610274910926819,0.02670419216156006,0.05935871601104736,0.9120283126831055,0.9088672399520874,0.6879287362098694,0.15197449922561646,0.5746928453445435,0.13426542282104492,0.9524457454681396,0.6333463788032532,0.21572434902191162,0.9370623230934143,0.9345961213111877};

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
  uint64_t d = 5;
  graph_writer_t writer;
  auto x = writer.input({d,d});
  auto y = writer.input(vector<uint64_t>{d,d});
  auto z = writer.ew(
    "ij,ij->ij",
    scalarop_t::make_mul(default_dtype()),
    writer.softmax(x), y);
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
  insert_data(data, graph, x.get_id());
  insert_data(data, graph, y.get_id());

  data = reference_compute_graph(graph, data);

  int gx_id = gx.get_id();
  int gy_id = gy.get_id();
  DOUT(data.at(z.get_id()));
  DOUT(gx_id << " gx " << data.at(gx_id).sum());
  DOUT(gy_id << " gy " << data.at(gy_id).sum());
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



int main() {
  set_default_dtype(dtype_t::f32);

  //rms_norm_test();
  //attention_test();

  complex_test();
  //complex_test2();
  //DOUT("")
  //complex_test3();

  //softmax_test();
  //reduction_test();
  //reduction_test2();
}
