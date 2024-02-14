//
// Created by Rich on 2021/1/23.
//

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "compressor.h"

CgrCompressor::CgrCompressor(int _zeta_k,
                             int _min_itv_len,
                             int _itv_seg_len,
                             int _res_seg_len,
                             int _one_way_storage)
    : zeta_k(_zeta_k), min_itv_len(_min_itv_len),
      itv_seg_len(_itv_seg_len), res_seg_len(_res_seg_len),
      node_num(0), edge_num(0), one_way_storage(_one_way_storage) {}

bool CgrCompressor::load_graph(const string &file_path)
{
    FILE *graph_file = fopen(file_path.c_str(), "r");
    if (graph_file == 0)
    {
        cout << "file cannot open!" << endl;
        abort();
    }

    size_type u = 0;
    size_type v = 0;
    while (fscanf(graph_file, "%ld %ld", &u, &v) > 0)
    {
        assert(u >= 0);
        assert(v >= 0);
        node_num = std::max(node_num, u + 1);
        node_num = std::max(node_num, v + 1);
        // if (!edges.empty())
        // {
        //     if (edges.back().first == u)
        //     {
        //         assert(edges.back().second < v);
        //     }
        // }
        edges.emplace_back(std::pair<size_type, size_type>(u, v));
    }
    edge_num = edges.size();
    adjlist.resize(node_num);
    for (auto edge : edges)
    {
        adjlist[edge.first].emplace_back(edge.second);
        if (one_way_storage)
        {
            adjlist[edge.second].emplace_back(edge.first);
        }
    }
    for (size_type i = 0; i < adjlist.size(); i++)
    {
        if (!adjlist[i].empty())
        {
            std::sort(adjlist[i].begin(), adjlist[i].end());
        }
    }
    fclose(graph_file);
    PRE_ENCODE_NUM = node_num + 5;
    return true;
}

void CgrCompressor::compress()
{
    pre_encoding();
    cgr.clear();
    cgr.resize(node_num);
    
    for (size_type i = 0; i < node_num; i++)
    {
        encode_node(i);
    }

    cout << "compress done" << endl;
}

// return the cost of compression in seconds
double CgrCompressor::compresswithcost()
{
    pre_encoding();
    cgr.clear();
    cgr.resize(node_num);
    // report compression cost 
    auto startofcompression =  std::chrono::high_resolution_clock::now();
    for (size_type i = 0; i < node_num; i++)
    {
        encode_node(i);
    }
    auto endofcompression =  std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endofcompression - startofcompression).count();
    cout << "compress done" << endl;
    return (double)duration/1000000;
}

void CgrCompressor::pre_encoding()
{
    gamma_code.clear();
    gamma_code.resize(PRE_ENCODE_NUM);
    zeta_code.clear();
    zeta_code.resize(PRE_ENCODE_NUM);
    for (size_type i = 0; i < PRE_ENCODE_NUM; i++)
    {
        encode_gamma(gamma_code[i], i);
        if (zeta_k == 1)
        {
            zeta_code[i] = gamma_code[i];
        }
        else
        {
            encode_zeta(zeta_code[i], i);
        }
    }
}

void CgrCompressor::encode_gamma(bits &bit_array, size_type x)
{
    x++;
    assert(x >= 0);
    int len = get_significant_bit(x); //获取x的有效长度
    encode(bit_array, 1, len + 1);
    encode(bit_array, x, len);
}

void CgrCompressor::encode_zeta(bits &bit_array, size_type x)
{
    if (zeta_k == 1)
    {
        encode_gamma(bit_array, x);
    }
    else
    {
        x++;
        assert(x >= 0);
        int len = get_significant_bit(x);
        int h = len / zeta_k;
        encode(bit_array, 1, h + 1);
        encode(bit_array, x, (h + 1) * zeta_k);
    }
}

void CgrCompressor::encode(bits &bit_array, size_type x, int len)
{
    for (int i = len - 1; i >= 0; i--)
    {
        bit_array.emplace_back((x >> i) & 1L);
    }
    return;
}

bool CgrCompressor::write_cgr(const string &file_path, int small_node_thresh, int large_node_thresh)
{
    bits graph;
    FILE *of_graph = fopen((file_path + ".graph").c_str(), "w");
    if (of_graph == 0)
    {
        cout << "graph file cannot create" << endl;
        abort();
    }
    FILE *of_offset = fopen((file_path + ".offset").c_str(), "w");
    if (of_offset == 0)
    {
        cout << "offset file cannot create!" << endl;
        abort();
    }
    int small_nb = small_node_thresh;
    cout << "compressor do not compress nodes whose nb number less than: " << small_nb << endl;
    write_bit_array(of_graph, of_offset, file_path, small_nb, large_node_thresh);
    return true;
}

void CgrCompressor::write_bit_array(FILE *&graph, FILE *&offset, const string &file_path,
                                    int small_node_nb, int large_node_nb)
{
    size_type count = 0;
    for (size_type i = 0; i < node_num; i++)
    {
        if (adjlist[i].size() >= small_node_nb)
        {
            count++;
        }
    }
    //offset首先会记录node的总数，读offset数组的时候需要注意
    fprintf(offset, "%ld\n", count);
    cout << "compress number of mid and large nodes are: " << count << endl;
    
    unsigned char cur = 0;
    vector<unsigned char> buf;
    size_type last_offset = 0;
    int big_nodes = 0;
    int bit_count = 0;
    //handle large nodes
    for (size_type i = 0; i < node_num; i++)
    {
        if (adjlist[i].size() >= large_node_nb)
        {
            big_nodes++;
            for (auto bit : cgr[i].bit_arr)
            {
                cur <<= 1;
                if (bit)
                {
                    cur++;
                }
                bit_count++;
                if (bit_count == 8)
                {
                    buf.emplace_back(cur);
                    cur = 0;
                    bit_count = 0;
                }
            }
            fprintf(offset, "%ld\n", cgr[i].bit_arr.size() + last_offset);
            last_offset += cgr[i].bit_arr.size();
            vertices.push_back(i);
        }
    }
    cout << "compress large nodes: " << big_nodes << endl;
    //handle mid nodes
    int mid_nodes = 0;
    for (size_type i = 0; i < node_num; i++)
    {
        if (adjlist[i].size() >= small_node_nb && adjlist[i].size() < large_node_nb)
        {
            mid_nodes++;
            for (auto bit : cgr[i].bit_arr)
            {
                cur <<= 1;
                if (bit)
                {
                    cur++;
                }
                bit_count++;
                if (bit_count == 8)
                {
                    buf.emplace_back(cur);
                    cur = 0;
                    bit_count = 0;
                }
            }

            fprintf(offset, "%ld\n", cgr[i].bit_arr.size() + last_offset);
            last_offset += cgr[i].bit_arr.size();
            vertices.push_back(i);
        }
    }
    cout << "compress mid nodes: " << mid_nodes << endl;
    if (bit_count)
    {
        while (bit_count < 8)
        {
            cur <<= 1, bit_count++;
        }
        buf.emplace_back(cur);
    }
   
    fwrite(buf.data(), sizeof(unsigned char), buf.size(), graph);
    fclose(graph);
    fclose(offset);

    // handle small nodes
    small_csr_offset.push_back(0);
    last_offset = 0;
    for (size_type i = 0; i < node_num; i++)
    {
        if (adjlist[i].size() < small_node_nb)
        {
            for (auto e : adjlist[i])
            {
                small_csr_graph.push_back(e);
            }
            vertices.push_back(i);
            small_csr_offset.push_back(last_offset + adjlist[i].size());
            last_offset += adjlist[i].size();
        }
    }
    write_vector2_binary(file_path + ".vertices.bin", vertices);
    if (!small_csr_graph.empty())
    {
        write_vector2_binary(file_path + ".graph.small.bin", small_csr_graph);
        write_vector2_binary(file_path + ".offset.small.bin", small_csr_offset);
    }
    return;
}

void CgrCompressor::write_vector2_binary(string file_path, vector<int> &data)
{
    FILE *f = fopen(file_path.c_str(), "w");

    if (f == 0)
    {
        cout << "file: " << file_path << " cannot create!" << std::endl;
        abort();
    }
    fwrite(data.data(), sizeof(int), data.size(), f);
    fclose(f);
    return;
}

void CgrCompressor::encode_node(const size_type node)
{
    auto &adj = cgr[node];
    adj.node = node;
    adj.outd = adjlist[node].size();
    adj.itv_left.clear();
    adj.itv_len.clear();
    adj.res.clear();
    adj.bit_arr.clear();

    if (res_seg_len == 0)
    {
        append_gamma(adj.bit_arr, adjlist[node].size());
        if (adjlist[node].empty())
        {
            return;
        }
    }
    intervalize(node);
    encode_intervals(node);
    encode_residuals(node);
    return;
}

void CgrCompressor::intervalize(const size_type node)
{
    size_type cur_left = 0, cur_right = 0;

    auto &neighbors = adjlist[node];
    auto &adj = cgr[node];

    while (cur_left < neighbors.size())
    {
        cur_right = cur_left + 1;
        while (cur_right < neighbors.size() &&
               neighbors[cur_right - 1] + 1 == neighbors[cur_right])
        {
            cur_right++;
        }
        size_type cur_len = cur_right - cur_left;

        if ((cur_len >= min_itv_len) && (min_itv_len != 0))
        {
            adj.itv_left.emplace_back(neighbors[cur_left]);
            adj.itv_len.emplace_back(cur_len);
        }
        else
        {
            for (size_type i = cur_left; i < cur_right; i++)
            {
                adj.res.emplace_back(neighbors[i]);
            }
        }
        cur_left = cur_right;
    }
    return;
}

void CgrCompressor::encode_intervals(const size_type node)
{
    auto &bit_arr = cgr[node].bit_arr;
    auto &itv_left = cgr[node].itv_left;
    auto &itv_len = cgr[node].itv_len;

    typedef std::pair<size_type, bits> segment;
    std::vector<segment> segs;

    bits cur_seg;
    size_type itv_cnt = 0;

    for (size_type i = 0; i < itv_left.size(); i++)
    {
        size_type cur_left = 0;
        if (itv_cnt == 0)
        {
            cur_left = int_2_nat(itv_left[i] - node);
        }
        else
        {
            cur_left = itv_left[i] - itv_left[i - 1] - itv_len[i - 1] - 1;
        }
        size_type cur_len = itv_len[i] - min_itv_len;

        // check if cur seg is overflowed
        if (itv_seg_len && gamma_size(itv_cnt + 1) + cur_seg.size() +
                                   gamma_size(cur_left) +
                                   gamma_size(cur_len) >
                               itv_seg_len)
        {
            segs.emplace_back(segment(itv_cnt, cur_seg));
            itv_cnt = 0;
            cur_left = int_2_nat(itv_left[i] - node);
            cur_seg.clear();
        }
        itv_cnt++;
        append_gamma(cur_seg, cur_left);
        append_gamma(cur_seg, cur_len);
    }
    // handle last paritial segment
    // 如果只有一个segment，那么这个segment也是在这里处理，在上面不会处理
    if (segs.empty())
    {
        //如果只有一个segment，直接push到segs

        segs.emplace_back(segment(itv_cnt, cur_seg));
    }
    else
    {
        //如果有多个segment，则把最后一个segment附加到倒数第二个上面。
        segs.back().first += itv_cnt;
        for (size_type i = itv_left.size() - itv_cnt; i < itv_left.size();
             i++)
        {
            // append_gamma x出现了负数
            append_gamma(segs.back().second, itv_left[i] - itv_left[i - 1] -
                                                 itv_len[i - 1] - 1);
            append_gamma(segs.back().second,
                         itv_len[i] - min_itv_len);
        }
    }

    if (itv_seg_len != 0)
    {
        //最终存储的位置还是bit_arr ,先存储多少个segment
        append_gamma(bit_arr, segs.size() - 1);
    }
    for (size_type i = 0; i < segs.size(); i++)
    {
        //再存储每个segment的内容，除了最后一块chunk（如果只有一块，那么也是第一块），都做长度为_itv_seg_len的对齐处理
        size_type align = i + 1 == segs.size() ? 0 : itv_seg_len;
        append_segment(bit_arr, segs[i].first, segs[i].second, align);
    }
    return;
}

void CgrCompressor::encode_residuals(const size_type node)
{
    auto &bit_arr = cgr[node].bit_arr;
    auto &res = cgr[node].res;

    typedef std::pair<size_type, bits> segment;
    std::vector<segment> segs;
    bits cur_seg;
    size_type res_cnt = 0;
    for (size_type i = 0; i < res.size(); i++)
    {
        size_type cur;
        if (res_cnt == 0)
        {
            cur = int_2_nat(res[i] - node);
        }
        else
        {
            cur = res[i] - res[i - 1] - 1;
        }
        // check if cur seg is overflowed
        if (res_seg_len && gamma_size(res_cnt + 1) + cur_seg.size() + zeta_size(cur) > res_seg_len)
        {
            segs.emplace_back(segment(res_cnt, cur_seg));
            res_cnt = 0;
            cur = int_2_nat(res[i] - node);
            cur_seg.clear();
        }
        res_cnt++;
        append_zeta(cur_seg, cur);
    }
    // handle last partial segment
    if (segs.empty())
    {
        segs.emplace_back(segment(res_cnt, cur_seg));
    }
    else
    {
        segs.back().first += res_cnt;
        for (size_type i = res.size() - res_cnt; i < res.size(); i++)
        {
            append_zeta(segs.back().second, res[i] - res[i - 1] - 1);
        }
    }
    if (res_seg_len != 0)
    {
        append_gamma(bit_arr, segs.size() - 1);
        for (size_type i = 0; i < segs.size(); i++)
        {
            size_type align = i + 1 == segs.size() ? 0 : res_seg_len;
            append_segment(bit_arr, segs[i].first, segs[i].second, align);
        }
    }
    else
    {
        bit_arr.insert(bit_arr.end(), cur_seg.begin(), cur_seg.end());
    }
    return;
}

void CgrCompressor::append_segment(bits &bit_array, size_type cnt,
                                   bits &cur_seg, size_type align)
{
    bits buf;
    append_gamma(buf, cnt);
    buf.insert(buf.end(), cur_seg.begin(), cur_seg.end());

    assert(align == 0 or buf.size() <= align);
    while (buf.size() < align)
    {
        buf.emplace_back(false);
    }
    bit_array.insert(bit_array.end(), buf.begin(), buf.end());
    return;
}

void CgrCompressor::append_gamma(bits &bit_array, size_type x)
{
    if (x < PRE_ENCODE_NUM)
    {
        bit_array.insert(bit_array.end(), gamma_code[x].begin(),
                         gamma_code[x].end());
    }
    else
    {
        encode_gamma(bit_array, x);
    }
    return;
}

void CgrCompressor::append_zeta(bits &bit_array, size_type x)
{
    if (x < PRE_ENCODE_NUM)
    {
        bit_array.insert(bit_array.end(), zeta_code[x].begin(),
                         zeta_code[x].end());
    }
    else
    {
        encode_zeta(bit_array, x);
    }
    return;
}