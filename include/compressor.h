//
// Created by Rich on 2021/1/23.
//

#ifndef LPA_COMPRESSOR_H
#define LPA_COMPRESSOR_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using size_type = int64_t;
using bits = std::vector<bool>;
// std
using std::cout;
using std::endl;
using std::pair;
using std::string;
using std::vector;

class CgrCompressor
{
public:
    explicit CgrCompressor(int _zeta_k = 3, int _min_itv_len = 4,
                           int _itv_seg_len = 0, int _res_seg_len = 4 * 32,
                           int _one_way = true);

    bool load_graph(const string &file_path);

    bool write_cgr(const string &file_path, int small_node_thresh = 32, int large_node_thresh = 128);

    void write_bit_array(FILE *&graph, FILE *&offset, const string &file_path,
                         int small_node_nb, int large_node_nb);

    void write_vector2_binary(string file_path, vector<int> &data);

    void encode_node(const size_type node);

    void intervalize(const size_type node);

    void encode_intervals(const size_type node);

    void encode_residuals(const size_type node);
    // 如果 align 不为0，则需要做对齐处理
    void append_segment(bits &bit_array, size_type cnt, bits &cur_seg,
                        size_type align);

    void append_gamma(bits &bit_array, size_type x);

    void append_zeta(bits &bit_array, size_type x);

    void compress();

    double compresswithcost();

    void pre_encoding();

    void encode_gamma(bits &bit_array, size_type x);

    void encode_zeta(bits &bit_array, size_type x);

    void encode(bits &bit_array, size_type x, int len);

    size_type int_2_nat(size_type x)
    {
        return x >= 0L ? x << 1 : -((x << 1) + 1L);
    }

    size_type gamma_size(size_type x)
    {
        if (x < this->PRE_ENCODE_NUM)
            return this->gamma_code[x].size();
        x++;
        assert(x >= 0);
        int len = this->get_significant_bit(x);
        return 2 * len + 1;
    }

    size_type zeta_size(size_type x)
    {
        if (x < this->PRE_ENCODE_NUM)
            return this->zeta_code[x].size();
        x++;
        assert(x >= 0);
        int len = this->get_significant_bit(x);
        int h = len / zeta_k;
        return (h + 1) * (zeta_k + 1);
    }

    int get_significant_bit(size_type x)
    {
        assert(x > 0);
        int ret = 0;
        while (x > 1)
            x >>= 1, ret++;
        return ret;
    }

private:
    size_type PRE_ENCODE_NUM = 200 * 100;
    int zeta_k;
    int min_itv_len;
    int itv_seg_len;
    int res_seg_len;
    size_type node_num;
    size_type edge_num;
    bool one_way_storage;
    vector<pair<size_type, size_type>> edges;
    vector<vector<size_type>> adjlist;
    vector<int> vertices;
    vector<int> small_csr_graph;
    vector<int> small_csr_offset;
    class CgrAdjlist
    {
    public:
        size_type node;
        size_type outd;
        vector<size_type> itv_left;
        vector<size_type> itv_len;
        vector<size_type> res;
        bits bit_arr;

        CgrAdjlist()
        {
            node = outd = 0;
            itv_left.clear();
            itv_len.clear();
            res.clear();
            bit_arr.clear();
        }
    };

    vector<CgrAdjlist> cgr;
    vector<bits> gamma_code;
    vector<bits> zeta_code;
};
#endif //LPA_COMPRESSOR_H
