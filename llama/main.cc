#include "misc.h"
#include "modules.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/gwriter.h"
#include "../src/einsummable/reference.h"
#include "../src/autoplace/alocate.h"
#include "../src/autoplace/apart.h"

#include <fstream>

using tensor_t = graph_writer_t::tensor_t;

void test_rms_norm() {
  dtype_t dtype = dtype_t::f16;
  set_default_dtype(dtype);

  auto [buffer_w, shape_w] = read_array(dtype,
    "[0.4661, 0.9641, 0.3547, 0.0875, 0.7970]");
  auto [buffer_x, shape_x] = read_array(dtype,
    "[[[0.0333, 0.0559, 0.7846, 0.7317, 0.6389],    "
    "     [0.5292, 0.5690, 0.5824, 0.7414, 0.8861], "
    "     [0.3719, 0.0415, 0.2344, 0.7994, 0.4197], "
    "     [0.3989, 0.5676, 0.8946, 0.9538, 0.0057]],"
    "    [[0.8891, 0.3468, 0.7371, 0.6944, 0.7015], "
    "     [0.9771, 0.7700, 0.5610, 0.5581, 0.0082], "
    "     [0.8135, 0.8535, 0.5942, 0.2557, 0.0224], "
    "     [0.4991, 0.5930, 0.1843, 0.4931, 0.5189]],"
    "    [[0.6224, 0.0205, 0.0650, 0.1405, 0.9574], "
    "     [0.3674, 0.7396, 0.6307, 0.2756, 0.8961], "
    "     [0.3821, 0.8173, 0.3014, 0.9887, 0.1508], "
    "     [0.1859, 0.1640, 0.6373, 0.1771, 0.4573]]]");
  auto [buffer_y, shape_y] = read_array(dtype,
    "[[[0.0277, 0.0965, 0.4977, 0.1145, 0.9106],    "
    "     [0.3654, 0.8128, 0.3061, 0.0961, 1.0464], "
    "     [0.3856, 0.0890, 0.1850, 0.1556, 0.7443], "
    "     [0.2808, 0.8266, 0.4793, 0.1260, 0.0068]],"
    "    [[0.5946, 0.4798, 0.3752, 0.0871, 0.8023], "
    "     [0.6906, 1.1259, 0.3018, 0.0740, 0.0099], "
    "     [0.6303, 1.3680, 0.3504, 0.0372, 0.0297], "
    "     [0.4856, 1.1936, 0.1365, 0.0900, 0.8635]],"
    "    [[0.5628, 0.0383, 0.0447, 0.0238, 1.4804], "
    "     [0.2736, 1.1393, 0.3574, 0.0385, 1.1411], "
    "     [0.2885, 1.2765, 0.1732, 0.1401, 0.1947], "
    "     [0.2302, 0.4203, 0.6007, 0.0412, 0.9684]]]");

  uint64_t d = 5;
  graph_writer_t writer;
  float eps = 1e-6;
  rms_norm_t rms_norm(&writer, "name", full_dim_t{ {d} }, eps);

  tensor_t x = writer.input(shape_x, dtype);
  tensor_t y = rms_norm.forward(x);

  y = y.save();

  map<int, dbuffer_t> inputs {
    { x.get_id(), buffer_x },
    { rms_norm.weight.get_id(), buffer_w }
  };

  auto saves = reference_compute_graph(writer.get_graph(), inputs);
  dbuffer_t _buffer_y = saves.at(y.get_id());

  DOUT(buffer_y);
  DOUT(_buffer_y);
}

void test_rotary_embedding() {
  string freqs_cis_str =
  " [[[ 1.000000000000,  0.000000000000],         "
  "          [ 1.000000000000,  0.000000000000],  "
  "          [ 1.000000000000,  0.000000000000]], "
  "         [[ 0.540302276611,  0.841470956802],  "
  "          [ 0.998923003674,  0.046399220824],  "
  "          [ 0.999997675419,  0.002154432703]], "
  "         [[-0.416146844625,  0.909297406673],  "
  "          [ 0.995694220066,  0.092698492110],  "
  "          [ 0.999990701675,  0.004308855161]]] ";
  string x_str =
  "[[[[0.660853981972, 0.514580428600, 0.666632235050, 0.181335031986,       "
  "           0.847351968288, 0.799827754498],                               "
  "          [0.271067082882, 0.889989793301, 0.746763706207, 0.133234560490,"
  "           0.746044099331, 0.387328267097],                               "
  "          [0.979579210281, 0.754689633846, 0.982744276524, 0.913159012794,"
  "           0.892353415489, 0.881933748722],                               "
  "          [0.136859238148, 0.099013566971, 0.662675976753, 0.375685930252,"
  "           0.136933088303, 0.758734345436]],                              "
  "                                                                          "
  "         [[0.508751749992, 0.929383039474, 0.079879045486, 0.775821268559,"
  "           0.480012178421, 0.453091800213],                               "
  "          [0.394689083099, 0.559428572655, 0.315627038479, 0.104672849178,"
  "           0.288223683834, 0.052234113216],                               "
  "          [0.831676244736, 0.879619419575, 0.715838313103, 0.053437113762,"
  "           0.415535748005, 0.316258013248],                               "
  "          [0.216684758663, 0.422591388226, 0.777273297310, 0.220147490501,"
  "           0.618135631084, 0.222365975380]],                              "
  "                                                                          "
  "         [[0.044190704823, 0.837435185909, 0.677218079567, 0.251019954681,"
  "           0.204477488995, 0.138726890087],                               "
  "          [0.345975339413, 0.755054414272, 0.588568985462, 0.510426819324,"
  "           0.444623351097, 0.419371187687],                               "
  "          [0.102480292320, 0.852209568024, 0.122825384140, 0.813726425171,"
  "           0.277293324471, 0.043915450573],                               "
  "          [0.365137755871, 0.344369769096, 0.445467650890, 0.923457801342,"
  "           0.546958386898, 0.677065014839]]],                             "
  "                                                                          "
  "                                                                          "
  "        [[[0.144875764847, 0.742299616337, 0.931350350380, 0.255407869816,"
  "           0.229083836079, 0.037463843822],                               "
  "          [0.217905879021, 0.171765446663, 0.163319647312, 0.043518126011,"
  "           0.572520911694, 0.917720317841],                               "
  "          [0.389085292816, 0.603349745274, 0.933605492115, 0.004259347916,"
  "           0.003709018230, 0.799461841583],                               "
  "          [0.012252330780, 0.573391020298, 0.760557115078, 0.302053391933,"
  "           0.343152642250, 0.831601321697]],                              "
  "                                                                          "
  "         [[0.307963967323, 0.089320480824, 0.896371662617, 0.274024546146,"
  "           0.737747848034, 0.944743275642],                               "
  "          [0.647053062916, 0.131549477577, 0.109595596790, 0.826745450497,"
  "           0.448201596737, 0.481170296669],                               "
  "          [0.452644407749, 0.025116741657, 0.783243060112, 0.359871089458,"
  "           0.294570326805, 0.781981348991],                               "
  "          [0.489667415619, 0.494699537754, 0.936191618443, 0.488254129887,"
  "           0.602399647236, 0.696564316750]],                              "
  "                                                                          "
  "         [[0.332710862160, 0.014498054981, 0.472303390503, 0.588377416134,"
  "           0.228377580643, 0.305674970150],                               "
  "          [0.855847716331, 0.568668246269, 0.393108785152, 0.620560288429,"
  "           0.077101528645, 0.124414205551],                               "
  "          [0.812509417534, 0.025072336197, 0.142413079739, 0.579947948456,"
  "           0.954405546188, 0.813515424728],                               "
  "          [0.161866784096, 0.482698917389, 0.708340764046, 0.495998740196,"
  "           0.071359515190, 0.515194475651]]]]                             ";
  string y_str =
  " [[[[ 6.608539819717e-01,  5.145804286003e-01,  6.666322350502e-01,          "
  "             1.813350319862e-01,  8.473519682884e-01,  7.998277544975e-01],  "
  "           [ 2.710670828819e-01,  8.899897933006e-01,  7.467637062073e-01,   "
  "             1.332345604897e-01,  7.460440993309e-01,  3.873282670975e-01],  "
  "           [ 9.795792102814e-01,  7.546896338463e-01,  9.827442765236e-01,   "
  "             9.131590127945e-01,  8.923534154892e-01,  8.819337487221e-01],  "
  "           [ 1.368592381477e-01,  9.901356697083e-02,  6.626759767532e-01,   "
  "             3.756859302521e-01,  1.369330883026e-01,  7.587343454361e-01]], "
  "                                                                             "
  "          [[-5.071690678596e-01,  9.302476048470e-01,  4.379551485181e-02,   "
  "             7.786920666695e-01,  4.790349006653e-01,  4.541248977184e-01],  "
  "           [-2.574914991856e-01,  6.343799233437e-01,  3.104303777218e-01,   "
  "             1.192049607635e-01,  2.881104648113e-01,  5.285494774580e-02],  "
  "           [-2.908176183701e-01,  1.175091743469e+00,  7.125878930092e-01,   "
  "             8.659390360117e-02,  4.148534238338e-01,  3.171525001526e-01],  "
  "           [-2.385231256485e-01,  4.106610119343e-01,  7.662215232849e-01,   "
  "             2.559752762318e-01,  6.176550984383e-01,  2.236971855164e-01]], "
  "                                                                             "
  "          [[-7.798674702644e-01, -3.083135187626e-01,  6.510329842567e-01,   "
  "             3.127162158489e-01,  2.038778364658e-01,  1.396066546440e-01],  "
  "           [-8.305456042290e-01,  3.809645713773e-04,  5.387189388275e-01,   "
  "             5.627884864807e-01,  4.428122043610e-01,  4.212830960751e-01],  "
  "           [-8.175587654114e-01, -2.614592611790e-01,  4.686531051993e-02,   "
  "             8.216084241867e-01,  2.771015167236e-01,  4.510985687375e-02],  "
  "           [-4.650854766369e-01,  1.887104213238e-01,  3.579464256763e-01,   "
  "             9.607757329941e-01,  5.440359115601e-01,  6.794154644012e-01]]],"
  "                                                                             "
  "                                                                             "
  "         [[[ 1.448757648468e-01,  7.422996163368e-01,  9.313503503799e-01,   "
  "             2.554078698158e-01,  2.290838360786e-01,  3.746384382248e-02],  "
  "           [ 2.179058790207e-01,  1.717654466629e-01,  1.633196473122e-01,   "
  "             4.351812601089e-02,  5.725209116936e-01,  9.177203178406e-01],  "
  "           [ 3.890852928162e-01,  6.033497452736e-01,  9.336054921150e-01,   "
  "             4.259347915649e-03,  3.709018230438e-03,  7.994618415833e-01],  "
  "           [ 1.225233078003e-02,  5.733910202980e-01,  7.605571150780e-01,   "
  "             3.020533919334e-01,  3.431526422501e-01,  8.316013216972e-01]], "
  "                                                                             "
  "          [[ 9.123303741217e-02,  3.074027895927e-01,  8.826917409897e-01,   "
  "             3.153203725815e-01,  7.357107400894e-01,  9.463304877281e-01],  "
  "           [ 2.389091849327e-01,  6.155528426170e-01,  7.111721485853e-02,   "
  "             8.309402465820e-01,  4.471639096737e-01,  4.821347892284e-01],  "
  "           [ 2.234297990799e-01,  3.944577574730e-01,  7.657017707825e-01,   "
  "             3.958253860474e-01,  2.928849160671e-01,  7.826141715050e-01],  "
  "           [-1.517068743706e-01,  6.793282032013e-01,  9.125287532806e-01,   "
  "             5.311668515205e-01,  6.008975505829e-01,  6.978605389595e-01]], "
  "                                                                             "
  "          [[-1.516396254301e-01,  2.964998185635e-01,  4.157280623913e-01,   "
  "             6.296257972717e-01,  2.270583510399e-01,  3.066561818123e-01],  "
  "           [-8.732468485832e-01,  5.415706038475e-01,  3.338911533356e-01,   "
  "             6.543288826942e-01,  7.656472921371e-02,  1.247452721000e-01],  "
  "           [-3.609214425087e-01,  7.283789515495e-01,  8.803957700729e-02,   "
  "             5.906522870064e-01,  9.508913755417e-01,  8.176202774048e-01],  "
  "           [-5.062772035599e-01, -5.368858203292e-02,  6.593124866486e-01,   "
  "             5.595251917839e-01,  6.913895159960e-02,  5.154972076416e-01]]]]";

  auto [_buffer_freqs_cis, shape_freqs_cis] = read_array(dtype_t::f32, freqs_cis_str);
  dbuffer_t buffer_freqs_cis(dtype_t::c64, _buffer_freqs_cis.data);
  shape_freqs_cis.resize(shape_freqs_cis.size()-1);
  shape_freqs_cis.back() /= 2;

  auto [buffer_x, shape_x] = read_array(dtype_t::f32, x_str);
  auto [buffer_y, shape_y] = read_array(dtype_t::f32, y_str);

  uint64_t db = shape_x[0];
  uint64_t dx = shape_x[1];
  uint64_t dc = shape_x[2];
  uint64_t dy = shape_x[3];

  graph_writer_t writer;
  tensor_t x = writer.input(shape_x, dtype_t::f32);
  tensor_t freqs_cis = writer.input({dx,dy/2}, dtype_t::c64);

  tensor_t y = attention_t::_apply_rotary_embedding(writer, x, freqs_cis);
  y = y.save();

  map<int, dbuffer_t> inputs {
    { x.get_id(), buffer_x },
    { freqs_cis.get_id(), buffer_freqs_cis }
  };

  auto saves = reference_compute_graph(writer.get_graph(), inputs);
  dbuffer_t _buffer_y = saves.at(y.get_id());

  if(!is_close(buffer_y, _buffer_y)) {
    throw std::runtime_error("NOT CLOSE");
  }
}

void test_ff() {
  string w1_str =
  "[[ 0.0199,  1.0192, -0.3516,  0.5422,  0.3592],"
  " [ 0.3241,  0.2817, -0.6044, -0.1984,  0.4958],"
  " [-0.4349,  0.2221,  0.1012, -0.2053, -0.0433],"
  " [ 0.8096,  0.3327, -0.1348, -0.0926,  0.8427],"
  " [ 0.6504, -0.0648,  0.9351, -0.3052, -0.4023],"
  " [ 0.2623, -0.1240,  0.2221, -0.2995, -0.2558]]";
  string w2_str =
  " [[ 0.7779,  0.8390,  0.3712, -0.0308, -0.4586, -0.1796],"
  "  [-0.3015,  0.5475,  0.0810, -0.7946, -0.0390,  0.2882],"
  "  [-0.4517, -0.5486,  0.6819, -0.0215,  0.5628,  0.8171],"
  "  [ 0.3962, -0.0222,  0.4035, -0.5755, -0.3978,  0.2763],"
  "  [-0.2128, -1.1187,  0.0275,  0.5822,  0.2103, -0.0682]]";
  string w3_str =
  " [[-1.2983, -0.6325,  0.0527, -0.2543, -0.2145],"
  "  [-1.2947, -0.3191,  0.4928,  0.4344,  0.0975],"
  "  [-0.3214,  0.3689, -0.3226,  0.1691,  0.2586],"
  "  [-0.5336, -0.1728, -0.0969, -0.1857,  0.4786],"
  "  [ 0.2574,  0.0682,  0.5974,  0.5393, -0.3378],"
  "  [ 0.1259, -0.0882, -0.0069,  0.4481, -0.3393]]";
  string x_str =
  "[[[0.8646, 0.0727, 0.1194, 0.3303, 0.1831], "
  "  [0.0827, 0.9898, 0.1074, 0.3576, 0.6346], "
  "  [0.6890, 0.5772, 0.4051, 0.2458, 0.5058], "
  "  [0.3608, 0.3060, 0.6976, 0.1480, 0.8217]],"
  " [[0.6197, 0.9016, 0.3980, 0.9648, 0.2971], "
  "  [0.2139, 0.3908, 0.8350, 0.8998, 0.1789], "
  "  [0.2042, 0.4409, 0.7889, 0.8857, 0.6012], "
  "  [0.7110, 0.1075, 0.8852, 0.7250, 0.7447]]]";
  string y_str =
  "[[[-0.3197,  0.2050,  0.2770,  0.0347,  0.0657], "
  "  [-0.8554,  0.2990,  0.5315, -0.4040,  0.2788], "
  "  [-0.7574,  0.3438,  0.4959, -0.2003,  0.2001], "
  "  [-0.2331,  0.0331,  0.1670, -0.1520,  0.0993]],"
  " [[-1.5784,  0.8233,  0.9322, -0.6242,  0.2730], "
  "  [-0.4669,  0.0813,  0.3672, -0.2431,  0.1922], "
  "  [-0.5078,  0.1742,  0.3255, -0.2471,  0.1495], "
  "  [-0.4784,  0.2611,  0.4380, -0.1997,  0.0522]]]";

  dtype_t dtype = dtype_t::f32;

  auto [buffer_w1, shape_w1] = read_array(dtype, w1_str);
  auto [buffer_w2, shape_w2] = read_array(dtype, w2_str);
  auto [buffer_w3, shape_w3] = read_array(dtype, w3_str);

  auto [buffer_x, shape_x] = read_array(dtype, x_str);
  auto [buffer_y, shape_y] = read_array(dtype, y_str);

  graph_writer_t writer;
  tensor_t x = writer.input(shape_x, dtype);

  uint64_t dim = shape_x.back();
  uint64_t hidden = shape_w1[0];

  feedforward_t ff(
    &writer,
    "name",
    full_dim_t::singleton(dim),
    hidden);
  tensor_t y = ff.forward(x);
  y = y.save();

  map<int, dbuffer_t> inputs {
    { x.get_id(), buffer_x },
    { ff.w1.get_id(), buffer_w1 },
    { ff.w2.get_id(), buffer_w2 },
    { ff.w3.get_id(), buffer_w3 }
  };

  auto saves = reference_compute_graph(writer.get_graph(), inputs);
  dbuffer_t _buffer_y = saves.at(y.get_id());

  if(!is_close(buffer_y, _buffer_y, 0.01)) {
    throw std::runtime_error("NOT CLOSE");
  }
}

int main() {
  set_default_dtype(dtype_t::f16);

  auto args = model_args_t::llama_7B();

  // TODO: set vocab_size
  args.vocab_size = 123;
  args.n_layers = 4;

  graph_writer_t writer;

  uint64_t bsz = 3;
  uint64_t seq_len = 17;

  args.batch_size = bsz;

  auto model = transformer_t(&writer, args, 0);

  // read bsz and seq_len and input input tensor
  auto input_input = [&]() {
    full_shape_t shape({
      full_dim_t::singleton(bsz),
      full_dim_t::singleton(seq_len),
      args.full_dim()
    });

    return writer.input(shape);
  };

  tensor_t x = input_input();
  tensor_t y = model.forward(x);

  //seq_len = 1;

  //x = input_input();
  //y = model.forward(x);

  y = y.save();
  
  graph_t const& graph = writer.get_graph();
  vector<partition_t> parts = apart01(graph,4);
  vector<placement_t> placements = alocate01(graph, parts,4,100);

  auto [inn_gid_to_tids, out_gid_to_tids, part_graph] = taskgraph_t::make(graph, placements);


  {
    std::ofstream f("llama_7B_4device.gv");
    part_graph.print_graphviz(f);
    DOUT("wrote to g.gv");
  }
}


