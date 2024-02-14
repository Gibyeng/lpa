#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "compressor.h"
#include "decompressor.cuh"
#include "file.h"
#include "utils.h"
#include "gtest/gtest.h"

using SIZE_TYPE = uint32_t;
const string file_path = "/home/yuchen/tmq/test_graph/pokec.edgelist";

void errorCheck(std::string message)
{
    auto err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        printf("Error! %s : %s\n", message.c_str(), cudaGetErrorString(err));
    }
}

__global__ void print_nb_of_node(SIZE_TYPE *row, SIZE_TYPE *clm, int node, int target)
{
    int begin = row[node];   // 3652
    int end = row[node + 1]; // 4647
    // target =8302
    // node=75
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x * gridDim.x)
    {
        if (clm[i] == target)
        {
            // i=3849
            printf("thread: %d find nb: %d of node: %d in clm[%d],begin:%d ,end:%d\n", threadIdx.x, target, node, i,
                   begin, end);
            break;
        }
    }
}

__global__ void decompress_large_vertex(int end, SIZE_TYPE *graph, SIZE_TYPE *vertices, SIZE_TYPE *offsets,
                                        SIZE_TYPE *row, SIZE_TYPE *clm, int *ret)
{
    int block_id = blockIdx.x;
    int steps = gridDim.x;
    for (int i = block_id; i < end; i += steps)
    {
        __shared__ typename Decompressor::SMem_b smem;
        SIZE_TYPE v = vertices[i];
        Decompressor decoder(&smem, graph, offsets, row, clm, ret, v, i);
        decoder.decompress();
    }
}

__global__ void decompress_mid_vertex(int begin, int end, SIZE_TYPE *graph, SIZE_TYPE *vertices, SIZE_TYPE *offset,
                                      SIZE_TYPE *row, SIZE_TYPE *clm, int *ret)
{
    int tid_in_all = threadIdx.x + blockIdx.x * blockDim.x;
    int steps = gridDim.x * (blockDim.x / 32);
    int warpid_in_all = (tid_in_all) / 32;
    int lane_id = threadIdx.x % 32;
    for (int i = warpid_in_all + begin; i < end; i += steps)
    {
        __shared__ typename Decompressor::SMem_b smem;
        SIZE_TYPE v = vertices[i];
        if (v == 14)
        {
            if (lane_id == 0)
            {
                // cgr_offset_index = 307000 ,cgr_offset= 0
                printf("node: %d,num_nb=%d, cgr_offset_index=%d, cgr_offset= %d\n", v, row[v + 1] - row[v], i,
                       offset[i]);
            }
        }
        Decompressor decoder(&smem, graph, offset, row, clm, ret, v, i);
        decoder.decompress();
    }
}

bool load_graph_and_test_decompress()
{

    File test_file = File(file_path, true, false, 32);
    test_file.load_graph();
    errorCheck("start test");
    // 二分查找大中小点的分解位置
    int right = test_file.row.size() - 1;
    // E E V V V
    int big_index = binary_search(SIZE_TYPE(0), SIZE_TYPE(right), test_file.row, test_file.vertices, SIZE_TYPE(128));
    int medium_index = binary_search(SIZE_TYPE(0), SIZE_TYPE(right), test_file.row, test_file.vertices, SIZE_TYPE(32));

    int num_big = big_index;
    int num_medium = medium_index - big_index;
    cout << "num_big: " << num_big << " num_mid: " << num_medium << endl;
    // large vertex 0- idx -1
    // 大点解压，每个点解压的邻居传回host，排序后与clm数组指定区域比较
    int block_num = num_big < 512 ? num_big : 512;
    cout << "block num of big nodes = " << block_num << endl;
    block_num = min(divup(num_medium, 4), 512);
    cout << "block num of mid nodes = " << block_num << endl;

    int tst = 1;
    int *h_test = &tst;
    int *d_test;
    cudaMalloc(&d_test, sizeof(int));
    cudaMemcpy(d_test, h_test, sizeof(int), cudaMemcpyHostToDevice);

    // graph data host2device
    SIZE_TYPE *d_graph;
    cudaMalloc(&d_graph, sizeof(SIZE_TYPE) * test_file.graph.size());
    cudaMemcpy(d_graph, &test_file.graph[0], sizeof(SIZE_TYPE) * test_file.graph.size(), cudaMemcpyHostToDevice);
    SIZE_TYPE *d_offset;
    cudaMalloc(&d_offset, sizeof(SIZE_TYPE) * test_file.offset.size());
    cudaMemcpy(d_offset, &test_file.offset[0], sizeof(SIZE_TYPE) * test_file.offset.size(), cudaMemcpyHostToDevice);
    SIZE_TYPE *d_vertices;
    cudaMalloc(&d_vertices, sizeof(SIZE_TYPE) * test_file.vertices.size());
    cudaMemcpy(d_vertices, &test_file.vertices[0], sizeof(SIZE_TYPE) * test_file.vertices.size(),
               cudaMemcpyHostToDevice);
    SIZE_TYPE *d_row;
    cudaMalloc(&d_row, sizeof(SIZE_TYPE) * test_file.row.size());
    cudaMemcpy(d_row, &test_file.row[0], sizeof(SIZE_TYPE) * test_file.row.size(), cudaMemcpyHostToDevice);
    SIZE_TYPE *d_clm;
    cudaMalloc(&d_clm, sizeof(SIZE_TYPE) * test_file.clm.size());
    cudaMemcpy(d_clm, &test_file.clm[0], sizeof(SIZE_TYPE) * test_file.clm.size(), cudaMemcpyHostToDevice);

    errorCheck("before decode big nodes");
    decompress_large_vertex<<<block_num, 128>>>(big_index, d_graph, d_vertices, d_offset, d_row, d_clm, d_test);

    cudaDeviceSynchronize();
    errorCheck("after decode big nodes");
    cudaMemcpy(h_test, d_test, sizeof(int), cudaMemcpyDeviceToHost);
    if (*h_test == 1)
    {
        cout << "big node decode test passed" << endl;
        // return true;
    }
    else
    {
        return false;
    }
    // 中点解压，每个点解压的邻居传回host，排序后与clm数组指定区域比较
    // mid vertex idx - right

    decompress_mid_vertex<<<block_num, 128>>>(big_index, medium_index - 1, d_graph, d_vertices, d_offset, d_row, d_clm,
                                              d_test);

    cudaDeviceSynchronize();
    errorCheck("after decode mid nodes");

    vector<int> nbs(test_file.clm.size());
    cudaMemcpy(h_test, d_test, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&nbs[0], d_clm, sizeof(int) * test_file.clm.size(), cudaMemcpyDeviceToHost);
    if (*h_test == 1)
    {
        cout << "mid node decode test passed" << endl;
        return true;
    }
    return false;
}

TEST(g_test, test1)
{
    // EXPECT_EQ (Decompressor::g_test(0),  0); //通过
    // EXPECT_EQ (Decompressor::g_test (2), 4); //通过
    EXPECT_EQ(load_graph_and_test_decompress(), true); //通过
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
