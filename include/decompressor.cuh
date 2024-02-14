//
// Created by Rich on 2021/1/25.
//

#ifndef LPA_DECOMPRESS_H
#define LPA_DECOMPRESS_H

//cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include "../3rd_party/cub/cub/cub.cuh"

#include "graph_utils.cuh"
#include <stdio.h>
#include <iostream>

using namespace cooperative_groups;
namespace cg = cooperative_groups;
const int THREADS_NUM = 128;

class Decompressor
{

public:
    typedef cub::BlockScan<SIZE_TYPE, THREADS_NUM> BlockScan;
    typedef cub::WarpScan<SIZE_TYPE> WarpScan;
    struct SMem_b
    {
        typename BlockScan::TempStorage block_temp_storage;
        typename WarpScan::TempStorage temp_storage[THREADS_NUM / 32];
        volatile SIZE_TYPE left[THREADS_NUM];
        volatile SIZE_TYPE len[THREADS_NUM];
        volatile SIZE_TYPE comm[THREADS_NUM / 32][32];
    };
    __device__
    Decompressor(SMem_b *_smem,
                 SIZE_TYPE *_graph,
                 SIZE_TYPE *_offset,
                 SIZE_TYPE *_row,
                 SIZE_TYPE *_clm,
                 int *test_flag,
                 int _node,
                 int _offset_index);
    Decompressor(){};

    __device__ void decompress();
    // __device__ bool test_decompress();

private:
    int *test_flag;
    SIZE_TYPE *row;
    SIZE_TYPE *clm;
    SIZE_TYPE *offsets;
    SIZE_TYPE *graph;
    SMem_b *smem;
    int node;
    int offset_index;
    int thread_id;
    int lane_id;
    int warp_id;
    int tid_in_all;
    __device__ void large_vertex_decompress();

    __device__ void large_vertex_handle_interval_segs(CgrReader &decoder);

    __device__ void large_vertex_handle_one_interval_seg(SIZE_TYPE v_node, volatile SIZE_TYPE &global_offset);

    __device__ void expand_interval(SIZE_TYPE left, SIZE_TYPE len);

    __device__ void expand_itv_more_than_128(SIZE_TYPE &left, SIZE_TYPE &len);

    __device__ void expand_itv_more_than_32(SIZE_TYPE &left, SIZE_TYPE &len);

    __device__ void expand_itv_more_than_0(SIZE_TYPE &left, SIZE_TYPE &len);

    __device__ void large_vertex_handle_residual_segs(CgrReader &decoder);

    __device__ void large_vertex_handle_one_residual_seg(SIZE_TYPE v_node, volatile SIZE_TYPE &global_offset);

    __device__ void medium_vertex_decompress();

    __device__ void medium_vertex_handle_interval_segs(CgrReader &decoder);

    __device__ void medium_vertex_handle_one_interval_seg(SIZE_TYPE v_node, volatile SIZE_TYPE &global_offset);

    __device__ void expand_m_interval(SIZE_TYPE left, SIZE_TYPE len);

    __device__ void expand_mitv_more_than_32(SIZE_TYPE &left, SIZE_TYPE &len);

    __device__ void expand_mitv_more_than_0(SIZE_TYPE &left, SIZE_TYPE &len);

    __device__ void medium_vertex_handle_residual_segs(CgrReader &decoder);

    __device__ void medium_vertex_handle_one_residual_seg(SIZE_TYPE v_node, volatile SIZE_TYPE &global_offset);
};

__device__
Decompressor::Decompressor(SMem_b *_smem,
                           SIZE_TYPE *_graph,
                           SIZE_TYPE *_offset,
                           SIZE_TYPE *_row,
                           SIZE_TYPE *_clm,
                           int *_test_flag,
                           int _node,
                           int _offset_index)
    : smem(_smem), graph(_graph), offsets(_offset), clm(_clm),
      row(_row), node(_node), offset_index(_offset_index), test_flag(_test_flag)
{
    thread_id = threadIdx.x;
    lane_id = thread_id % 32;
    warp_id = thread_id / 32;
    tid_in_all = threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ void Decompressor::decompress()
{
    if (row[node + 1] - row[node] >= THREADS_NUM)
    {
        large_vertex_decompress();
    }
    else
    {

        medium_vertex_decompress();
    }
    __syncthreads();

    if (thread_id == 0)
    {
        int begin = row[node];
        int end = row[node + 1];
        for (int i = begin; i < end; i++)
        {
            if (clm[i] != 0 && node < 100)
            {
                printf("node:%d missed its %dth nb: %d \n", node, i - begin, clm[i]);
                *test_flag = -1;
            }
        }
    }

    return;
}

__device__ void Decompressor::large_vertex_decompress()
{
    CgrReader decoder;
    SIZE_TYPE row_begin = offsets[offset_index];
    decoder.init(node, graph, row_begin);
    large_vertex_handle_interval_segs(decoder);
    large_vertex_handle_residual_segs(decoder);
    return;
}

__device__ void Decompressor::large_vertex_handle_interval_segs(CgrReader &decoder)
{
    SIZE_TYPE itv_segment_cnt = decoder.decode_segment_cnt();
    if (itv_segment_cnt == 0)
    {
        return;
    }
    int block_progress = 0;
    int seg_idx = thread_id;
    SIZE_TYPE begin = decoder.global_offset;
    while (block_progress < itv_segment_cnt)
    {
        volatile SIZE_TYPE offset = begin + seg_idx * 256;
        SIZE_TYPE v_node = node;
        if (seg_idx >= itv_segment_cnt)
        {
            v_node = SIZE_NONE;
        }
        large_vertex_handle_one_interval_seg(v_node, offset);

        // determine which thread handle the last itv seg
        // sync
        __syncthreads();
        if (seg_idx == itv_segment_cnt - 1)
        {
            smem->comm[0][0] = offset;
        }
        __syncthreads();

        seg_idx += THREADS_NUM;
        block_progress += THREADS_NUM;
    }
    decoder.global_offset = smem->comm[0][0];
}

__device__ void Decompressor::large_vertex_handle_one_interval_seg(SIZE_TYPE v_node, volatile SIZE_TYPE &global_offset)
{
    CgrReader decoder;
    decoder.init(v_node, graph, global_offset);
    IntervalSegmentHelper sh(v_node, decoder);
    sh.decode_interval_cnt();
    SIZE_TYPE thread_data = sh.interval_cnt;
    SIZE_TYPE rsv_rank;
    SIZE_TYPE total;
    __syncthreads();
    BlockScan(smem->block_temp_storage)
        .ExclusiveSum(thread_data, rsv_rank, total);
    __syncthreads();
    SIZE_TYPE cta_progress = 0;
    SIZE_TYPE remain;
    while (cta_progress < total)
    {
        smem->len[thread_id] = 0;
        __syncthreads();
        remain = total - cta_progress;
        while ((rsv_rank < cta_progress + THREADS_NUM) && (sh.interval_cnt))
        {
            SIZE_TYPE left = sh.get_interval_left();
            SIZE_TYPE len = sh.get_interval_len();
            smem->left[rsv_rank - cta_progress] = left;
            smem->len[rsv_rank - cta_progress] = len;
            rsv_rank++;
        }
        __syncthreads();
        SIZE_TYPE left = 0, len = 0;
        if (thread_id < min(remain, THREADS_NUM))
        {
            left = smem->left[thread_id];
            len = smem->len[thread_id];
        }
        expand_interval(left, len);
        cta_progress += THREADS_NUM;
        __syncthreads();
    }
    // compute the global offset
    global_offset = decoder.global_offset;
}

__device__ void Decompressor::expand_interval(SIZE_TYPE left, SIZE_TYPE len)
{
    expand_itv_more_than_128(left, len);
    expand_itv_more_than_32(left, len);
    expand_itv_more_than_0(left, len);
    return;
}

__device__ void Decompressor::expand_itv_more_than_128(SIZE_TYPE &left, SIZE_TYPE &len)
{
    //如果某个len长度大于thread num
    //则block中全部的thread都去处理这个 <left,len>
    while (__syncthreads_or(len >= THREADS_NUM))
    {
        // vie for control of block
        if (len >= THREADS_NUM)
        {
            smem->comm[0][0] = thread_id;
        }
        __syncthreads();
        // winner describes adjlist
        if (smem->comm[0][0] == thread_id)
        {
            smem->comm[0][1] = left;
            smem->comm[0][2] = left + len / THREADS_NUM * THREADS_NUM;
            left += len - len % THREADS_NUM;
            len %= THREADS_NUM;
        }
        __syncthreads();
        // gather = neighbour
        SIZE_TYPE gather = smem->comm[0][1] + thread_id;

        SIZE_TYPE gather_end = smem->comm[0][2];
        SIZE_TYPE tmp_left = smem->comm[0][1];
        SIZE_TYPE tmp_right = smem->comm[0][2];
        if (thread_id == 0)
        {
            SIZE_TYPE len = tmp_right - tmp_left;

            int begin = row[node];
            int end = row[node + 1];
            for (int i = begin; i < end; i++)
            {
                if (clm[i] == tmp_left)
                {
                    for (int j = i; j < i + len; j++)
                    {
                        if (clm[j + 1] != clm[j] + 1)
                        {
                            *test_flag = -1;
                            printf("[itv>128,b]decode wrong left: %d,right: %d of node: %d\n", tmp_left, tmp_right, node);
                            break;
                        }
                        clm[j] = 0;
                    }
                    break;
                }
            }
        }
    }
}

__device__ void Decompressor::expand_itv_more_than_32(SIZE_TYPE &left, SIZE_TYPE &len)
{

    while (__any_sync(FULL_MASK, len >= 32))
    {
        // vie for control of warp
        if (len >= 32)
        {
            smem->comm[warp_id][0] = lane_id;
        }
        // winner describes adjlist
        if (smem->comm[warp_id][0] == lane_id)
        {
            smem->comm[warp_id][1] = left;
            smem->comm[warp_id][2] = left + len / 32 * 32;
            left += len - len % 32;
            len %= 32;
        }

        SIZE_TYPE tmp_left = smem->comm[warp_id][1];
        SIZE_TYPE tmp_right = smem->comm[warp_id][2];
        if (lane_id == 0)
        {
            int begin = row[node];
            int end = row[node + 1];
            for (int i = begin; i < end; i++)
            {
                int len = tmp_right - tmp_left;
                if (clm[i] == tmp_left)
                {
                    for (int j = i; j < i + len - 1; j++)
                    {
                        if (clm[j + 1] != clm[j] + 1)
                        {
                            *test_flag = -1;
                            printf("[itv>32,b]decode wrong left: %d,right: %d of node: %d\n", tmp_left, tmp_right, node);
                            break;
                        }
                        clm[j] = 0;
                    }
                    clm[i + len - 1] = 0;
                    break;
                }
            }
        }
    }
}

__device__ void Decompressor::expand_itv_more_than_0(SIZE_TYPE &left, SIZE_TYPE &len)
{
    SIZE_TYPE thread_data = len;
    SIZE_TYPE rsv_rank;
    SIZE_TYPE total;
    SIZE_TYPE remain;
    SIZE_TYPE cnt = 0;
    __syncthreads();
    BlockScan(smem->block_temp_storage)
        .ExclusiveSum(thread_data, rsv_rank, total);
    __syncthreads();
    SIZE_TYPE cta_progress = 0;
    auto block = cg::this_thread_block();
    auto g = cg::tiled_partition<32>(block);
    while (cta_progress < total)
    {
        remain = total - cta_progress;
        while ((rsv_rank < cta_progress + THREADS_NUM) && (cnt < len))
        {
            smem->left[rsv_rank - cta_progress] = left++;
            rsv_rank++;
            cnt++;
        }
        __syncthreads();
        SIZE_TYPE neighbour = SIZE_NONE;
        if (thread_id < min(remain, THREADS_NUM))
        {
            neighbour = smem->left[thread_id];
            bool find = false;
            int begin = row[node];
            int end = row[node + 1];
            for (int i = begin; i < end; i++)
            {
                if (clm[i] == neighbour)
                {
                    clm[i] = 0;
                    find = true;
                    break;
                }
            }
            if (!find && node < 100)
            {
                printf("[itv>0,b]thread: %d decode wrong nb: %d of node: %d\n", thread_id, neighbour, node);
                *test_flag = -1;
            }
        }

        cta_progress += THREADS_NUM;
        __syncthreads();
    }
}

__device__ void Decompressor::large_vertex_handle_residual_segs(CgrReader &decoder)
{
    SIZE_TYPE res_segment_cnt = decoder.decode_segment_cnt();
    int block_progress = 0;
    int seg_idx = thread_id;
    while (block_progress < res_segment_cnt)
    {
        volatile SIZE_TYPE offset = decoder.global_offset + seg_idx * 256;
        SIZE_TYPE v_node = node;
        if (seg_idx >= res_segment_cnt)
        {
            v_node = SIZE_NONE;
        }
        large_vertex_handle_one_residual_seg(v_node, offset);
        // sync
        __syncthreads();
        seg_idx += THREADS_NUM;
        block_progress += THREADS_NUM;
    }
}

__device__ void Decompressor::large_vertex_handle_one_residual_seg(SIZE_TYPE v_node,
                                                                   volatile SIZE_TYPE &global_offset)
{
    CgrReader decoder;
    decoder.init(v_node, graph, global_offset);
    ResidualSegmentHelper sh(v_node, decoder);
    sh.decode_residual_cnt();
    SIZE_TYPE thread_data = sh.residual_cnt;
    SIZE_TYPE total;
    //        SIZE_TYPE warp_progress = 0;
    SIZE_TYPE remain;
    SIZE_TYPE rsv_rank;
    __syncthreads();
    BlockScan(smem->block_temp_storage)
        .ExclusiveSum(thread_data, rsv_rank, total);
    __syncthreads();
    SIZE_TYPE cta_progress = 0;
    while (cta_progress < total)
    {
        remain = total - cta_progress;
        while (rsv_rank < cta_progress + THREADS_NUM && sh.residual_cnt)
        {
            // 这里code出来了负数

            SIZE_TYPE neighbour = sh.get_residual();
            int left_index = rsv_rank - cta_progress;
            // 有两个线程同时解压出来了nb=8302
            // thread: 5 of block: 5,compressor decode nb: 8302,put it in smem left[59]
            // thread: 59 of block: 5,compressor got nb: 8302,check clm array from:3652 to 4647
            // thread: 1 of block: 5,compressor decode nb: 8302,put it in smem left[64]
            // thread: 64 of block: 5,compressor got nb: 8302,check clm array from:3652 to 4647
            // fixed:显然上述解压过程进行了两次，是由于上层调用函数的循环中，seg_idx 和 block_progress增量错误导致。
            smem->left[left_index] = neighbour;
            rsv_rank++;
        }
        __syncthreads();
        SIZE_TYPE neighbour = SIZE_NONE;
        if (thread_id < min(remain, THREADS_NUM))
        {
            neighbour = smem->left[thread_id];

            bool find = false;
            int begin = row[node];
            int end = row[node + 1];
            for (int i = begin; i < end; i++)
            {
                if (clm[i] == neighbour)
                {
                    clm[i] = 0;
                    find = true;
                    break;
                }
            }
            if (!find && node < 100)
            {
                *test_flag = -1;
                printf("[res,b]thread: %d decode wrong nb: %d of node: %d\n", thread_id, neighbour, node);
            }
        }
        cta_progress += THREADS_NUM;
        __syncthreads();
    }
}

__device__ void Decompressor::medium_vertex_decompress()
{
    CgrReader decoder;
    SIZE_TYPE row_begin = offsets[offset_index];
    decoder.init(node, graph, row_begin);

    medium_vertex_handle_interval_segs(decoder);
    medium_vertex_handle_residual_segs(decoder);
    __syncthreads();
}

__device__ void Decompressor::medium_vertex_handle_interval_segs(CgrReader &decoder)
{
    SIZE_TYPE itv_segment_cnt = decoder.decode_segment_cnt();
    if (itv_segment_cnt == 0)
    {
        return;
    }
    // each thread handle one segement
    int warp_progress = 0;
    int seg_idx = lane_id;
    while (warp_progress < itv_segment_cnt)
    {
        volatile SIZE_TYPE offset = decoder.global_offset + seg_idx * 256;
        SIZE_TYPE v_node = node;
        if (seg_idx >= itv_segment_cnt)
        {
            v_node = SIZE_NONE;
        }
        medium_vertex_handle_one_interval_seg(v_node, offset);
        // sync
        if (seg_idx == itv_segment_cnt - 1)
        {
            smem->comm[warp_id][0] = offset;
        }
        seg_idx += 32;
        warp_progress += 32;
        //这里需要这个同步吗？ warp级同步即可？
        __syncthreads();
    }
    // compute the write offset to decompress res nbs
    decoder.global_offset = smem->comm[warp_id][0];
}

__device__ void Decompressor::medium_vertex_handle_one_interval_seg(SIZE_TYPE v_node,
                                                                    volatile SIZE_TYPE &global_offset)
{
    CgrReader decoder;
    decoder.init(v_node, graph, global_offset);
    IntervalSegmentHelper sh(v_node, decoder);
    sh.decode_interval_cnt();
    SIZE_TYPE thread_data = sh.interval_cnt;
    SIZE_TYPE rsv_rank;
    SIZE_TYPE total;
    // sync?
    __syncthreads();

    WarpScan(smem->temp_storage[warp_id])
        .ExclusiveSum(thread_data, rsv_rank, total);
    __syncthreads();

    SIZE_TYPE warp_progress = 0;
    while (warp_progress < total)
    {
        int remain = total - warp_progress;
        while ((rsv_rank < warp_progress + 32) && (sh.interval_cnt))
        {
            smem->left[warp_id * 32 + rsv_rank - warp_progress] =
                sh.get_interval_left();
            smem->len[warp_id * 32 + rsv_rank - warp_progress] =
                sh.get_interval_len();
            rsv_rank++;
        }
        SIZE_TYPE left = 0, len = 0;
        if (lane_id < min(remain, 32))
        {
            left = smem->left[thread_id];
            len = smem->len[thread_id];
        }
        expand_m_interval(left, len);
        warp_progress += 32;
        __syncthreads();
    }
    // recompute global offset
    global_offset = decoder.global_offset;
}

__device__ void Decompressor::expand_m_interval(SIZE_TYPE left, SIZE_TYPE len)
{
    while (__any_sync(FULL_MASK, len >= 32))
    {
        // vie for control of warp
        if (len >= 32)
        {
            smem->comm[warp_id][0] = lane_id;
        }

        // winner describes adjlist
        if (smem->comm[warp_id][0] == lane_id)
        {
            smem->comm[warp_id][1] = left;
            smem->comm[warp_id][2] = left + len / 32 * 32;
            left += len - len % 32;
            len %= 32;
        }
        SIZE_TYPE tmp_left = smem->comm[warp_id][1];
        SIZE_TYPE tmp_right = smem->comm[warp_id][2];
        if (thread_id == 0)
        {
            int begin = row[node];
            int end = row[node + 1];
            SIZE_TYPE len = tmp_right - tmp_left;
            for (int i = begin; i < end; i++)
            {
                if (clm[i] == tmp_left)
                {
                    for (int j = i; j < i + len; j++)
                    {
                        if (clm[j + 1] != clm[j] + 1)
                        {
                            *test_flag = -1;
                            printf("[itv>32,m] decode wrong left: %d,right: %d of node: %d\n", tmp_left, tmp_right, node);
                            break;
                        }
                        clm[j] = 0;
                    }
                    break;
                }
            }
        }
    }
    // less than 32
    SIZE_TYPE thread_data = len;
    SIZE_TYPE rsv_rank;
    SIZE_TYPE total;
    SIZE_TYPE remain;
    WarpScan(smem->temp_storage[warp_id])
        .ExclusiveSum(thread_data, rsv_rank, total);
    SIZE_TYPE warp_progress = 0;
    int count = 0;
    while (warp_progress < total)
    {
        //
        remain = total - warp_progress;

        while ((rsv_rank < warp_progress + 32) && count < len) //?
        {
            smem->left[warp_id * 32 + rsv_rank - warp_progress] = left + count++;
            rsv_rank++;
        }
        SIZE_TYPE neighbour = SIZE_NONE;
        if (lane_id < min(remain, 32))
        {
            neighbour = smem->left[thread_id];
            bool find = false;
            int begin = row[node];
            int end = row[node + 1];
            for (int i = begin; i < end; i++)
            {
                if (clm[i] == neighbour)
                {
                    clm[i] = 0;
                    find = true;
                    break;
                }
            }
            if (!find && node < 100)
            {
                printf("[itv>0,m]thread: %d decode wrong nb: %d of node: %d\n", thread_id, neighbour, node);
                *test_flag = -1;
            }
        }
        warp_progress += 32;
    }
}

__device__ void Decompressor::medium_vertex_handle_residual_segs(CgrReader &decoder)
{
    SIZE_TYPE res_segment_cnt = decoder.decode_segment_cnt();
    // each thread handle one segment
    int warp_progress = 0;
    int seg_idx = lane_id;
    while (warp_progress < res_segment_cnt)
    {
        volatile SIZE_TYPE offset = decoder.global_offset + seg_idx * 256;
        SIZE_TYPE v_node = node;
        if (seg_idx >= res_segment_cnt)
            v_node = SIZE_NONE;
        medium_vertex_handle_one_residual_seg(v_node, offset);
        // sync
        __syncthreads();
        seg_idx += 32;
        warp_progress += 32;
    }
}

__device__ void Decompressor::medium_vertex_handle_one_residual_seg(SIZE_TYPE v_node,
                                                                    volatile SIZE_TYPE &global_offset)
{
    CgrReader decoder;
    decoder.init(v_node, graph, global_offset);
    ResidualSegmentHelper sh(v_node, decoder);
    sh.decode_residual_cnt();
    SIZE_TYPE thread_data = sh.residual_cnt;
    SIZE_TYPE rsv_rank;
    SIZE_TYPE total;
    SIZE_TYPE remain;
    WarpScan(smem->temp_storage[warp_id])
        .ExclusiveSum(thread_data, rsv_rank, total);
    SIZE_TYPE warp_progress = 0;
    while (warp_progress < total)
    {
        remain = total - warp_progress;
        while ((rsv_rank < warp_progress + 32) && (sh.residual_cnt))
        {
            smem->left[warp_id * 32 + rsv_rank - warp_progress] = sh.get_residual();
            rsv_rank++;
        }
        SIZE_TYPE neighbour = SIZE_NONE;
        if (lane_id < min(remain, 32))
        {
            neighbour = smem->left[thread_id];
            int begin = row[node];
            int end = row[node + 1];
            bool find = false;
            for (int i = begin; i < end; i++)
            {
                if (clm[i] == neighbour)
                {
                    clm[i] = 0;
                    find = true;
                    break;
                }
            }
            if (!find && node < 100)
            {
                printf("[res,m]thread: %d decode wrong nb: %d of node: %d\n", thread_id, neighbour, node);
                *test_flag = -1;
            }
        }
        warp_progress += 32;
    }
}

#endif //LPA_DECOMPRESS_H
