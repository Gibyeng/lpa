//
// Created by Rich on 2021/1/24.
//

#include "file.h"
#include "compressor.h"
#include "utils.h"

File::File(const string _file_name, bool _compress, bool _one_way, int sms)
    : file_name(_file_name), compress(_compress), one_way(_one_way),
      small_node_thresh(sms)
{
    has_small = sms == 0 ? false : true;
    cout << "图文件： " << _file_name << endl;
    cout << "是否压缩： " << _compress << endl;
    cout << "是否有小图： " << has_small << endl;
    if (has_small)
    {
        cout << "小点不压缩参数：" << sms << endl;
    }
    cout << "图源文件是否单向存储边： " << _one_way << endl;
    cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
}
void File::load_graph()
{
    cout << "load graph : " << file_name << endl;
    string csr_graph = file_name + ".clm";
    string csr_offset = file_name + ".row";
    if (compress)
    {
        //  如果是压缩图，则需要检查 .graph .offset .row 文件是否存在
        string cgr_graph = file_name + ".graph";
        string cgr_offset = file_name + ".offset";
        if (!file_exists(cgr_graph.c_str()) ||
            !file_exists(cgr_offset.c_str()) ||
            !file_exists(csr_offset.c_str()))
        {
            // 如果有其中一个不存在，重新生成
            write_binary();
        }
        load_compress_graph();
        // 压缩图需要重新加载排序后的节点排列
        load_vector_from_binary(file_name + ".vertices.bin", vertices);
    }
    else
    {
        if (!file_exists(csr_graph.c_str()))
        {
            write_binary();
        }
        load_vector_from_binary(csr_graph, clm);
    }

    // graph and offset for all nodes in csr format
    load_vector_from_binary(csr_offset, row);
    // 为了跑decode nb的测试，需要clm数组，所以这里打开了，如果内存紧张可注销掉
    load_vector_from_binary(csr_graph, clm);
    // graph and offsets for small nodes
    if (has_small)
    {
        load_vector_from_binary(file_name + ".graph.small.bin", small_graph);
        load_vector_from_binary(file_name + ".offset.small.bin", small_offset);
    }
    // node num total
    if (compress)
    {
        vertex_num = vertices.size();
    }
    else
    {
        vertex_num = row.size() - 1;
    }
    return;
}

void File::write_binary()
{
    read_from_edge_list();
    if (compress)
    {
        cout << "compressor start to compress: " << file_name << endl;
        Timer t;
        t.start();
        auto compressor = CgrCompressor(3, 4, 8 * 32, 8 * 32, one_way);
        compressor.load_graph(file_name);
        auto cost = compressor.compresswithcost();
        compressor.write_cgr(file_name, small_node_thresh, 128);
        t.stop();
        cout << "compress file: " << file_name << " elapse(with I/O): " << t.elapsed_second() << "without I/O: " << cost << endl;

    }
    write_vector2_binary(file_name + ".row", row);
    write_vector2_binary(file_name + ".clm", clm);
    return;
}

void File::load_vector_from_binary(string file_path, vector<Uint> &data)
{
    uint64_t FileSize = file_size(file_path.c_str());
    // load graph
    std::ifstream ifs;
    ifs.open(file_path, std::ios::in | std::ios::binary);

    if (!ifs.is_open())
    {
        std::cout << "open: " << file_path << "  failed!" << std::endl;
        abort();
    }
    ifs.seekg(0, std::ios::beg);
    data.resize(FileSize / sizeof(Uint));
    ifs.read((char *)data.data(), FileSize);
    ifs.close();
}

void File::read_from_edge_list()
{
    vector<vector<int>> adjList;
    FILE *f = fopen(file_name.c_str(), "r");
    if (f == 0)
    {
        cout << "file: " << file_name << " cannot open!" << endl;
        abort();
    }
    unsigned int u = 0;
    unsigned int v = 0;
    unsigned int max_u_v = 0;
    unsigned int cnt = 0;
    Timer t1;
    t1.start();
    while (fscanf(f, "%d %d", &u, &v) > 0)
    {
        if (adjList.size() <= max(u, v))
        {
            adjList.resize(max(u, v) + 1);
        }
        adjList[u].push_back(v);
        if (one_way)
        {
            adjList[v].push_back(u);
        }

		if(max_u_v < u|| max_u_v < v){
			max_u_v = max(u,v);
		}
    }
    t1.stop();
    cout << "read edgelist elapse: " << t1.elapsed_second() << endl;
    //邻接表 -> csr
    vector<Uint> _row;
    vector<Uint> _clm;
    Timer t2;
    t2.start();
    _row.push_back(0);
    // if(one_way){
        for (auto i = 0; i < adjList.size(); i++)
        {
            if (!adjList[i].empty())
            {
                std::sort(adjList[i].begin(), adjList[i].end());
                _clm.insert(_clm.end(), adjList[i].begin(), adjList[i].end());
            }
            _row.push_back(_clm.size());
        }
    // }else{
    //     auto size_accumulator = [](const E &a, decltype(adjList[0]) &b) {
    //         return a + b.size();
    //     };
    //     unsigned int m = std::accumulate(adjList.begin(), adjList.end(), E(), size_accumulator);
    //     _clm.resize(this->m);
    //     for (auto i = 0; i< max_u_v; ++i) {
    //         auto u = i;
    //         bool f = false;
    //         for (auto v: adjList[u]) {
    //             if (f && u < v) {
    //                 // // Insert a loop
    //                 // neighbors[cur++] = u;
    //                 f = false;
    //             }
    //             _clm[cur++] = v;
    //         }
    //         if (f) {
    //             _clm[cur++] = u;
    //         }
    //         _row[u + 1] = cur;
    //         // adjList[u].clear();
    //     }
    // }
    row.swap(_row);
    clm.swap(_clm);
    t2.stop();
    cout << "make offsets and neighbours elapse: " << t2.elapsed_second() << endl;
    cout << "read file from edgelists,clm_size: " << clm.size()
         << "; row_size: " << row.size() << endl;
    return;
}

void File::load_compress_graph()
{
    // load graph
    std::ifstream ifs;
    cout << "loading compress graph: " << file_name << ".graph" << endl;
    ifs.open(file_name + ".graph",
             std::ios::in | std::ios::binary | std::ios::ate);

    if (!ifs.is_open())
    {
        cout << "open: " << file_name << ".graph file failed!" << endl;
        abort();
    }

    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);

    ifs.read((char *)buffer.data(), size);

    graph.clear();
    Uint tmp = 0; // ??
    for (size_t i = 0; i < buffer.size(); i++)
    {
        tmp <<= 8;
        tmp += buffer[i];
        if ((i + 1) % 4 == 0)
        {
            graph.push_back(tmp);
        }
    }

    if (size % 4)
    {
        int rem = size % 4;
        while (rem % 4)
        {
            tmp <<= 8, rem++;
        }
        graph.push_back(tmp);
    }
    ifs.close();

    // load offset
    Uint num_node;
    offset.clear();
    offset.push_back(0); //?
    std::ifstream ifs_offset;
    ifs_offset.open(file_name + ".offset", std::ios::in);

    if (!ifs_offset.is_open())
    {
        cout << "open: " << file_name << ".offset failed" << endl;
        abort();
    }
    ifs_offset >> num_node;
    Uint cur;
    for (auto i = 0; i < num_node; i++)
    {
        ifs_offset >> cur;
        offset.push_back(cur);
    }
    ifs_offset.close();
    return;
}

void File::write_vector2_binary(string file_path, vector<Uint> &data)
{
    FILE *f = fopen(file_path.c_str(), "w");

    if (f == 0)
    {
        cout << "file: " << file_path << " cannot create!" << endl;
        abort();
    }
    fwrite(data.data(), sizeof(Uint), data.size(), f);
    fclose(f);
    return;
}
