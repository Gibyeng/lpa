#include <iostream>
#include <string>

#include "file.h"
#include "graph.h"
#include "incore_li.cuh"
#include "parse_command.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

const string dblp = "/home/yuchen/tmq/dataset/dblp/com-dblp.ungraph.txt";
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
    // pokec 是双向存储的，testgraph 是单向存储的
    bool compress = false;
    bool edge_storeage_once = false; // false ->双向存储 ，不需要再将 u添加至v
    int small_node_thresh = 0;
    std::string inputfile;
    std::string outputfile;
    std::ofstream writefile;
    if(argc == 3){
	std::cout << "读取输入参数..."<< std::endl;
    	inputfile = argv[1];   
	    outputfile =  argv[2];
    }else{
    	inputfile = dblp;
    }
    File f = File(inputfile, compress, edge_storeage_once, small_node_thresh);
    f.load_graph();
    cout << "Input graph check:" << endl;
    cout << "clm_size(csr_all_nodes): " << f.clm.size() << endl;
    cout << "row_size(csr_all_nodes): " << f.row.size() << endl;
    cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
    using Vertex = uint32_t;
    using Edge = uint32_t;
    using Propagator = LabelPropagator<Vertex, Edge>;
    using GraphT = CSRGraph<Vertex, Edge>;
    // Load a graph
    std::shared_ptr<GraphT> graph(new CSRGraph<Vertex, Edge>(f.clm, f.row));
    std::unique_ptr<Propagator> propagator;

    propagator = make_unique_ptr<InCoreLIDPP<Vertex, Edge>>(graph);
    std::pair<double, double> result = propagator->run(15);
    double f1 = result.first;
    double f2 = result.second;
    cout << "f1: " << f1 << ",f2: " << f2 << endl;
    if(argc != 1){
        writefile.open(outputfile,std::ios::app);
            writefile << "gsort file: " << inputfile <<std::endl<< " running time: " << f1/1000 << "thre: " << small_node_thresh <<std::endl;
        writefile <<"~~~~~"<< std::endl;        
        writefile.close();
    }
    return 0;
}