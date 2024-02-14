#include "cglp_propagator.cuh"
#include "file.h"
#include "graph.h"
#include "label_propagator.h"
#include "parse_command.h"
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::string;
using std::vector;

const string dblp = "../../../dataset/dblp/com-dblp.ungraph.txt";
const string pokec = "/home/yuchen/tmq/test_graph/pokec.edgelist";
const string roadnet = "/home/yuchen/tmq/dataset/roadnet/roadNet-CA.txt";
const string youtube = "/home/yuchen/tmq/dataset/youtube/com-youtube.ungraph.txt";
const string journal = "/home/yuchen/tmq/dataset/ljournal/soc-LiveJournal1.txt";
const string uk2002 = "/home/yuchen/tmq/dataset/uk-2002/uk-2002.txt";

int main(int argc, char *argv[])
{

    //    commandLine cmd(argc, argv, "<source-file>  [-c] compress  [-s] one-way storage [-n] uncompress small node ");
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
    //    File f = File(string(iFile), compressed, one_way, has_small, small_node_thresh);
    bool compress = true;
    bool edge_storeage_once = false;
    int small_node_thresh = 32;
    std::string inputfile;
    std::string outputfile;
    std::ofstream writefile;

    if(argc == 6){
	std::cout << "读取输入参数..."<< std::endl;
    	inputfile = argv[1];
        compress = (bool)atoi(argv[2]);
        edge_storeage_once = (bool)atoi(argv[3]);
	small_node_thresh = atoi(argv[4]);
	outputfile =  argv[5];
    }else{
    	inputfile = dblp;
    }
    // pokec 是双向存储的，testgraph 是单向存储的
    File f = File(inputfile, compress, edge_storeage_once, small_node_thresh);
    f.load_graph();
    cout << "Input graph check:" << endl;
    cout << "clm_size(csr_all_nodes): " << f.clm.size() << endl;
    cout << "row_size(csr_all_nodes): " << f.row.size() << endl;
    cout << "graph_size(cgr_big+mid_nodes): " << f.graph.size() << endl;
    cout << "offset_size(cgr_big+mid_nodes): " << f.offset.size() << endl;
    cout << "vertices size(all_nodes): " << f.vertices.size() << endl;

    cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
    using Vertex = uint32_t;
    using Edge = uint32_t;
    using Propagator = LabelPropagator<Vertex, Edge>;
    using GraphT = CGRGraphPure<Vertex, Edge>;
    // Load a graph
    std::shared_ptr<GraphT> graph(new CGRGraphPure<Vertex, Edge>(f.clm, f.row, f.graph, f.offset, f.vertices));
    std::unique_ptr<Propagator> propagator;

    propagator = make_unique_ptr<CGLPPropagator<Vertex, Edge>>(graph);
    std::pair<double, double> result = propagator->run(15);
    double f1 = result.first;
    double f2 = result.second;
    //output file
    if(argc != 1){
	writefile.open(outputfile,std::ios::app);
        writefile << "cglp file: " << inputfile <<std::endl<< " running time: " << f1/1000 << "thre: " << small_node_thresh <<std::endl;
	writefile <<"~~~~~"<< std::endl;        
	writefile.close();
    }
    cout << "f1: " << f1 << ",f2: " << f2 << endl;
    return 0;
}
