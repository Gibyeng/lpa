//
// Created by Rich on 2021/1/24.
//

#ifndef LPA_FILE_H
#define LPA_FILE_H

#include "compressor.h"
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using std::cin;
using std::cout;
using std::endl;
using std::max;
using std::string;
using std::vector;

using Uint = uint32_t;

class File
{
public:
    File(const string _file_name, bool _compress, bool _one_way, int sms = 32);

    void load_graph();
    // csr
    vector<Uint> clm;
    vector<Uint> row;
    // cgr
    vector<Uint> graph;
    vector<Uint> offset;
    vector<Uint> vertices;
    // 部分压缩时，需要存储小点的 csr
    vector<Uint> small_graph;
    vector<Uint> small_offset;
    Uint vertex_num;
    int small_node_thresh;

private:
    string file_name;
    bool compress;
    bool one_way;
    bool has_small;

    void read_from_edge_list();

    void write_binary();

    void load_compress_graph();

    void write_vector2_binary(string file_path, vector<Uint> &data);

    void load_vector_from_binary(string file_path, vector<Uint> &data);

    inline bool file_exists(const char *file_name)
    {
        struct stat buffer;
        return (stat(file_name, &buffer) == 0);
    }

    inline uint64_t file_size(const char *file_name)
    {
        struct stat st;
        stat(file_name, &st);
        uint64_t size = st.st_size;
        return size;
    }
};

#endif //LPA_FILE_H
