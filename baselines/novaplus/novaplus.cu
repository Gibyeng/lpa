#include "file.h"
#include "graph.h"
#include "novaplus_propogator.cuh"
#include "parse_command.h"
#include <fstream>
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::string;
using std::vector;

const string dblp =
    "/home/yuchen/tmq/dataset/dblp/com-dblp.ungraph.txt"; // true
const string pokec = "/home/yuchen/tmq/test_graph/pokec.edgelist";
const string roadnet =
    "/home/yuchen/tmq/dataset/roadnet/roadNet-CA.txt"; // false
const string youtube =
    "/home/yuchen/tmq/dataset/youtube/com-youtube.ungraph.txt"; // true
const string journal =
    "/home/yuchen/tmq/dataset/ljournal/soc-LiveJournal1.txt";         // false
const string uk2002 = "/home/yuchen/tmq/dataset/uk-2002/uk-2002.txt"; // true

const string test1 = "../../../dataset/dblp/dblp0/com-dblp.ungraph.txt";
const string test2 =
    "../../../dataset/roadnet/roadnet0/com-roadnet.ungraph.txt";

int main(int argc, char *argv[]) {

  //    commandLine cmd(argc, argv, "<source-file>  [-c] compress  [-s] one-way
  //    storage [-n] uncompress small node ");
  //    // handle command line args
  //    char *iFile = cmd.getArgument(0);
  //    bool compressed = cmd.getOptionValue("-c");
  //    bool one_way = cmd.getOptionValue("-s");
  //    int small_node_thresh = atoi(cmd.getOptionValue("-n"));
  //    cout <<"small node thresh: " << small_node_thresh<<endl;
  //    cout    <<"compress graph? :" <<compressed <<endl;
  //    cout    <<"source graph one-way storage? : "<<one_way<<endl;
  //    cout    <<"file path:" << string(iFile) << endl;
  //    bool has_small = small_node_thresh==0? false:true;
  //    // load graph
  //    File f = File(string(iFile), compressed, one_way, has_small,
  //    small_node_thresh);
  bool compress = true;
  bool edge_storeage_once = true;
  int small_node_thresh = 32;
  std::string inputfile;
  std::string outputfile;
  // pokec 是双向存储的，testgraph 是单向存储的
  // add argc;
  std::ofstream writefile;
  if (argc == 6) {
    std::cout << "读取输入参数..." << std::endl;
    inputfile = argv[1];
    compress = (bool)atoi(argv[2]);
    edge_storeage_once = (bool)atoi(argv[3]);
    small_node_thresh = atoi(argv[4]);
    outputfile = argv[5];
  } else {
    inputfile = test1;
    compress = true;
    edge_storeage_once = true;
    small_node_thresh = 0;
    outputfile = "./output.txt";
  }
  File f = File(inputfile, compress, edge_storeage_once, small_node_thresh);
  f.load_graph();
  cout << "Input graph check:" << endl;
  cout << "clm_size(csr_all_nodes): " << f.clm.size() << endl;
  cout << "row_size(csr_all_nodes): " << f.row.size() << endl;
  cout << "graph_size(cgr_big+mid_nodes): " << f.graph.size() << endl;
  cout << "offset_size(cgr_big+mid_nodes): " << f.offset.size() << endl;

  cout << "vertices size(all_nodes): " << f.vertices.size() << endl;
  cout << "small graph length: " << f.small_graph.size() << endl;
  cout << "small offset length: " << f.small_offset.size() << endl;
  cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
  using Vertex = uint32_t;
  using Edge = uint32_t;
  using Propagator = LabelPropagator<Vertex, Edge>;
  using GraphT = CGRGraph<Vertex, Edge>;
  // Load a graph
  std::shared_ptr<GraphT> graph(new CGRGraph<Vertex, Edge>(
      f.clm, f.row, f.graph, f.offset, f.small_graph, f.small_offset,
      f.vertices, small_node_thresh));
  std::unique_ptr<Propagator> propagator;

  propagator = make_unique_ptr<NovaPlusPropagator<Vertex, Edge>>(graph);
  std::pair<double, double> result = propagator->run(20);
  double f1 = result.first;
  double f2 = result.second;
  // f1: lp elapsed, f2: preprocess + lp
  cout << "f1: " << f1 / 1000 << "(s)"
       << ",f2: " << f2 << endl;
  // output file
  if (argc != 1) {
    writefile.open(outputfile, std::ios::app);
    writefile << "nova file: " << inputfile << std::endl
              << " running time: " << f1 / 1000 << "thre: " << small_node_thresh
              << std::endl;
    writefile << "~~~~~" << std::endl;
    writefile.close();
  }
  return 0;
}
