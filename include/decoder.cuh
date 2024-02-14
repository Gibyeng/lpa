//
// Created by Rich on 2021/1/25.
//

#pragma once

// cuda
#include "../3rd_party/cub/cub/cub.cuh"
#include <cooperative_groups.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cms.cuh"
#include "graph_utils.cuh"

#include <iostream>
#include <stdio.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

template <int w, int d, int htW, int THREADS_NUM = 128> class Decoder {
public:
  typedef cub::BlockScan<SIZE_TYPE, THREADS_NUM> BlockScan;
  typedef cub::WarpScan<SIZE_TYPE> WarpScan;
  struct SMem_b {
    typename BlockScan::TempStorage block_temp_storage;
    typename WarpScan::TempStorage temp_storage[THREADS_NUM / 32];
    volatile SIZE_TYPE left[THREADS_NUM];
    volatile SIZE_TYPE len[THREADS_NUM];
    volatile SIZE_TYPE comm[THREADS_NUM / 32][32];
    // for large vertex
    SharedCM<w, d> s_cms;
    // for mid vertex
    SIZE_TYPE vals[THREADS_NUM / 32][htW];
    SIZE_TYPE keys[THREADS_NUM / 32][htW];
  };
  Decoder(){};
  __device__ Decoder(SMem_b *_smem, SIZE_TYPE *_graph, SIZE_TYPE *_offsets,
                     SIZE_TYPE *_row, int *_labels, int _node,
                     int _offset_index, SIZE_TYPE _N)
      : smem(_smem), graph(_graph), offsets(_offsets), row(_row), node(_node),
        labels(_labels), offset_index(_offset_index), N(_N) {
    thread_id = threadIdx.x;
    lane_id = thread_id % 32;
    warp_id = thread_id / 32;
    tid_in_all = threadIdx.x + blockIdx.x * blockDim.x;
  }
  __device__ void decompress_and_lp() {
    if (row[node + 1] - row[node] >= THREADS_NUM) {
      large_vertex_decompress_and_lp();
    } else {
      medium_vertex_decompress_and_lp_basic();
      //   medium_vertex_decompress_and_lp_opt();
    }
    return;
  }
  __device__ void decompress_and_lp_basic() {
    if (row[node + 1] - row[node] >= THREADS_NUM) {
      // big node > THREADS_NUM can not be handle by shared memory
      large_vertex_decompress_and_lp();
    } else {
      // handle by shared memory
      medium_vertex_decompress_and_lp_basic();
    }
    return;
  }

  __device__ void decompress_and_lp_opt() {
    if (row[node + 1] - row[node] >= THREADS_NUM) {
      large_vertex_decompress_and_lp();
    } else {
      medium_vertex_decompress_and_lp_opt();
    }
    return;
  }

private:
  SIZE_TYPE *row;
  SIZE_TYPE *clm;
  SIZE_TYPE *offsets;
  SIZE_TYPE *graph;
  SMem_b *smem;
  int *labels;
  int node;
  int offset_index;
  int thread_id;
  int lane_id;
  int warp_id;
  int tid_in_all;
  // number of vertices
  int N;

  __device__ int hash(int k) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return abs(k);
  }
  __device__ void large_vertex_decompress_and_lp() {
    CgrReader reader;
    SIZE_TYPE row_begin = offsets[offset_index];
    int my_max_key = -1;
    int my_max_count = 0;
    reader.init(node, graph, row_begin);
    large_vertex_handle_interval_segs(reader, my_max_key, my_max_count);
    // illegal memory access
    large_vertex_handle_residual_segs(reader, my_max_key, my_max_count);
    __syncthreads();
    typedef cub::BlockReduce<unsigned long long, THREADS_NUM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    unsigned long long valkey;
    valkey = my_max_key == -1
                 ? 0
                 : ((unsigned long long)my_max_count << 32) + my_max_key;
    unsigned long long aggregate =
        BlockReduce(temp_storage).Reduce(valkey, cub::Max());
    __syncthreads();

    if (threadIdx.x == 0) {
      int lb = aggregate & 0xffffffff;
      labels[node] = lb;
      // printf("thread: %d update node: %d to label: %d\n", thread_id, node,
      // lb);
    }
    return;
  }

  __device__ void large_vertex_handle_interval_segs(CgrReader &reader,
                                                    int &my_max_key,
                                                    int &my_max_count) {
    SIZE_TYPE itv_segment_cnt = reader.decode_segment_cnt();
    if (itv_segment_cnt == 0) {
      return;
    }
    int block_progress = 0;
    int seg_idx = thread_id;
    SIZE_TYPE begin = reader.global_offset;
    while (block_progress < itv_segment_cnt) {
      volatile SIZE_TYPE offset = begin + seg_idx * 256;
      SIZE_TYPE v_node = node;
      if (seg_idx >= itv_segment_cnt) {
        v_node = SIZE_NONE;
      }
      large_vertex_handle_one_interval_seg(v_node, offset, my_max_key,
                                           my_max_count);

      // determine which thread handle the last itv seg
      // sync
      __syncthreads();
      if (seg_idx == itv_segment_cnt - 1) {
        smem->comm[0][0] = offset;
      }
      __syncthreads();

      seg_idx += THREADS_NUM;
      block_progress += THREADS_NUM;
    }
    reader.global_offset = smem->comm[0][0];
  }

  __device__ void
  large_vertex_handle_one_interval_seg(SIZE_TYPE v_node,
                                       volatile SIZE_TYPE &global_offset,
                                       int &my_max_key, int &my_max_count) {
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
    while (cta_progress < total) {
      smem->len[thread_id] = 0;
      __syncthreads();
      remain = total - cta_progress;
      while ((rsv_rank < cta_progress + THREADS_NUM) && (sh.interval_cnt)) {
        SIZE_TYPE left = sh.get_interval_left();
        SIZE_TYPE len = sh.get_interval_len();
        smem->left[rsv_rank - cta_progress] = left;
        smem->len[rsv_rank - cta_progress] = len;
        rsv_rank++;
      }
      __syncthreads();
      SIZE_TYPE left = 0, len = 0;
      if (thread_id < min(remain, THREADS_NUM)) {
        left = smem->left[thread_id];
        len = smem->len[thread_id];
      }
      expand_interval(left, len, my_max_key, my_max_count);
      cta_progress += THREADS_NUM;
      __syncthreads();
    }
    // compute the global offset
    global_offset = decoder.global_offset;
  }

  __device__ void expand_interval(SIZE_TYPE left, SIZE_TYPE len,
                                  int &my_max_key, int &my_max_count) {
    expand_itv_more_than_128(left, len, my_max_key, my_max_count);
    expand_itv_more_than_32(left, len, my_max_key, my_max_count);
    expand_itv_more_than_0(left, len, my_max_key, my_max_count);
    return;
  }

  __device__ void expand_itv_more_than_128(SIZE_TYPE &left, SIZE_TYPE &len,
                                           int &my_max_key, int &my_max_count) {
    // 如果某个len长度大于thread num
    // 则block中全部的thread都去处理这个 <left,len>
    auto block = cg::this_thread_block();
    auto g = cg::tiled_partition<32>(block);

    while (__syncthreads_or(len >= THREADS_NUM)) {
      // vie for control of block
      if (len >= THREADS_NUM) {
        smem->comm[0][0] = thread_id;
      }
      __syncthreads();
      // winner describes adjlist
      if (smem->comm[0][0] == thread_id) {
        smem->comm[0][1] = left;
        smem->comm[0][2] = left + len / THREADS_NUM * THREADS_NUM;
        left += len - len % THREADS_NUM;
        len %= THREADS_NUM;
      }
      __syncthreads();
      // gather = neighbour
      SIZE_TYPE gather = smem->comm[0][1] + thread_id;

      SIZE_TYPE gather_end = smem->comm[0][2];
      // 一次while循环中，每个thread都将处理同一个node的128个邻居中的一个
      while (__syncthreads_or(gather < gather_end)) {
        int key = labels[gather % N];
        auto lmask = __match_any_sync(FULL_MASK, key);
        auto lleader = __ffs(lmask) - 1;
        auto count = __popc(lmask);
        // leader insert to cms
        if (g.thread_rank() == lleader && key != 0) {
          int cm_count = smem->s_cms.increment(key, count);
          if (my_max_count < cm_count) {
            my_max_count = cm_count;
            my_max_key = key;
          }
        }
        gather += THREADS_NUM;
      }
    }
  }

  __device__ void expand_itv_more_than_32(SIZE_TYPE &left, SIZE_TYPE &len,
                                          int &my_max_key, int &my_max_count) {
    auto block = cg::this_thread_block();
    auto g = cg::tiled_partition<32>(block);
    while (__any_sync(FULL_MASK, len >= 32)) {
      // vie for control of warp
      if (len >= 32) {
        smem->comm[warp_id][0] = lane_id;
      }
      // winner describes adjlist
      if (smem->comm[warp_id][0] == lane_id) {
        smem->comm[warp_id][1] = left;
        smem->comm[warp_id][2] = left + len / 32 * 32;
        left += len - len % 32;
        len %= 32;
      }

      SIZE_TYPE gather = smem->comm[warp_id][1] + lane_id;
      SIZE_TYPE gather_end = smem->comm[warp_id][2];
      while (__any_sync(FULL_MASK, gather < gather_end)) {
        int mylanekey = -1;
        if (gather < gather_end)
          mylanekey = labels[gather % N];
        //                printf("thead: %d get itv-32 nb:
        //                %d\n",thread_id,gather);
        auto lmask = __match_any_sync(FULL_MASK, mylanekey);
        auto lleader = __ffs(lmask) - 1;
        auto count = __popc(lmask);
        if (g.thread_rank() == lleader && mylanekey != -1) {
          auto cm_count = smem->s_cms.increment(mylanekey, count);
          if (my_max_count < cm_count) {
            my_max_count = cm_count;
            my_max_key = mylanekey;
          }
        }
        gather += 32;
      }
    }
  }

  __device__ void expand_itv_more_than_0(SIZE_TYPE &left, SIZE_TYPE &len,
                                         int &my_max_key, int &my_max_count) {
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
    while (cta_progress < total) {
      remain = total - cta_progress;
      while ((rsv_rank < cta_progress + THREADS_NUM) && (cnt < len)) {
        smem->left[rsv_rank - cta_progress] = left++;
        rsv_rank++;
        cnt++;
      }
      __syncthreads();
      SIZE_TYPE neighbour = SIZE_NONE;
      int mylanekey = -1;
      if (thread_id < min(remain, THREADS_NUM)) {
        neighbour = smem->left[thread_id];
        mylanekey = labels[neighbour % N];
      }
      auto gmask = __match_any_sync(FULL_MASK, mylanekey);
      int count = 0;
      if (g.thread_rank() == __ffs(gmask) - 1 && mylanekey != -1) {
        count = __popc(gmask);
        int cm_count = smem->s_cms.increment(mylanekey, count);
        if (my_max_count < cm_count) {
          my_max_count = cm_count;
          my_max_key = mylanekey;
        }
      }
      cta_progress += THREADS_NUM;
      __syncthreads();
    }
  }

  __device__ void large_vertex_handle_residual_segs(CgrReader &reader,
                                                    int &my_max_key,
                                                    int &my_max_count) {

    SIZE_TYPE res_segment_cnt = reader.decode_segment_cnt();
    int block_progress = 0;
    int seg_idx = thread_id; // 没必要 warp - res_seg
    while (block_progress < res_segment_cnt) {
      volatile SIZE_TYPE offset = reader.global_offset + seg_idx * 256;
      SIZE_TYPE v_node = node;
      if (seg_idx >= res_segment_cnt) {
        v_node = SIZE_NONE;
      }
      large_vertex_handle_one_residual_seg(v_node, offset, my_max_key,
                                           my_max_count);
      // sync
      __syncthreads();
      seg_idx += THREADS_NUM;
      block_progress += THREADS_NUM;
    }
  }
  // one thread one segment
  __device__ void
  large_vertex_handle_one_residual_seg(SIZE_TYPE v_node,
                                       volatile SIZE_TYPE &global_offset,
                                       int &my_max_key, int &my_max_count) {
    CgrReader reader;
    reader.init(v_node, graph, global_offset);
    ResidualSegmentHelper sh(v_node, reader);
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
    auto block = cg::this_thread_block();
    auto g = cg::tiled_partition<32>(block);
    while (cta_progress < total) {
      remain = total - cta_progress;
      while (rsv_rank < cta_progress + THREADS_NUM && sh.residual_cnt) {
        // 这里code出来了负数
        SIZE_TYPE neighbour = sh.get_residual();

        smem->left[rsv_rank - cta_progress] = neighbour;

        rsv_rank++;
      }
      __syncthreads();
      SIZE_TYPE neighbour = SIZE_NONE;
      int mylanekey = -1;
      if (thread_id < min(remain, THREADS_NUM)) {
        neighbour = smem->left[thread_id];
        // illegical memory access
        neighbour %= N;
        mylanekey = labels[neighbour];
      }

      auto lmask = __match_any_sync(FULL_MASK, mylanekey);
      auto lleader = __ffs(lmask) - 1;
      auto count = __popc(lmask);
      // leader insert to cms
      if (g.thread_rank() == w && mylanekey != -1) {
        // printf("mylanekey \n", mylanekey);
        auto cm_count = smem->s_cms.increment(mylanekey, count);
        if (my_max_count < cm_count) {
          my_max_count = cm_count;
          my_max_key = mylanekey;
        }
      }
      cta_progress += THREADS_NUM;
      __syncthreads();
    }
  }

  __device__ void medium_vertex_decompress_and_lp() {
    int my_max_key = -1;
    int my_max_count = 0;
    CgrReader reader;
    SIZE_TYPE row_begin = offsets[offset_index];
    reader.init(node, graph, row_begin);

    medium_vertex_handle_interval_segs(reader, my_max_key, my_max_count);
    medium_vertex_handle_residual_segs(reader, my_max_key, my_max_count);
    __syncthreads();
    // warp reduce
    __syncwarp(FULL_MASK);
    typedef cub::WarpReduce<unsigned long long> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[THREAD_NUM / 32];
    unsigned long long valkey;
    valkey = my_max_key == -1
                 ? 0
                 : ((unsigned long long)my_max_count << 32) + my_max_key;
    unsigned long long aggregate =
        WarpReduce(temp_storage[warp_id]).Reduce(valkey, cub::Max());
    auto block = cg::this_thread_block();
    auto g = cg::tiled_partition<32>(block);
    if (g.thread_rank() == 0) {
      my_max_key = aggregate & 0xffffffff;
      labels[node] = my_max_key;
      // printf("thread: %d update node: %d to label: %d\n", thread_id, node,
      // my_max_key);
    }
  }

  __device__ void medium_vertex_handle_interval_segs(CgrReader &reader,
                                                     int &my_max_key,
                                                     int &my_max_count) {
    SIZE_TYPE itv_segment_cnt = reader.decode_segment_cnt();
    if (itv_segment_cnt == 0) {
      return;
    }
    // each thread handle one segement
    int warp_progress = 0;
    int seg_idx = lane_id;
    while (warp_progress < itv_segment_cnt) {
      volatile SIZE_TYPE offset = reader.global_offset + seg_idx * 256;
      SIZE_TYPE v_node = node;
      if (seg_idx >= itv_segment_cnt) {
        v_node = SIZE_NONE;
      }
      medium_vertex_handle_one_interval_seg(v_node, offset, my_max_key,
                                            my_max_count);
      // sync
      if (seg_idx == itv_segment_cnt - 1) {
        smem->comm[warp_id][0] = offset;
      }
      seg_idx += 32;
      warp_progress += 32;
      // 这里需要这个同步吗？ warp级同步即可？
      __syncthreads();
    }
    // compute the write offset to decompress res nbs
    reader.global_offset = smem->comm[warp_id][0];
  }

  __device__ void
  medium_vertex_handle_one_interval_seg(SIZE_TYPE v_node,
                                        volatile SIZE_TYPE &global_offset,
                                        int &my_max_key, int &my_max_count) {
    CgrReader reader;
    reader.init(v_node, graph, global_offset);
    IntervalSegmentHelper sh(v_node, reader);
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
    while (warp_progress < total) {
      int remain = total - warp_progress;
      while ((rsv_rank < warp_progress + 32) && (sh.interval_cnt)) {
        smem->left[warp_id * 32 + rsv_rank - warp_progress] =
            sh.get_interval_left();
        smem->len[warp_id * 32 + rsv_rank - warp_progress] =
            sh.get_interval_len();
        rsv_rank++;
      }
      SIZE_TYPE left = 0, len = 0;
      if (lane_id < min(remain, 32)) {
        left = smem->left[thread_id];
        len = smem->len[thread_id];
      }
      expand_m_interval(left, len, my_max_key, my_max_count);
      warp_progress += 32;
      __syncthreads();
    }
    // recompute global offset
    global_offset = reader.global_offset;
  }

  __device__ void expand_m_interval(SIZE_TYPE left, SIZE_TYPE len,
                                    int &my_max_key, int &my_max_count) {
    auto block = cg::this_thread_block();
    auto g = cg::tiled_partition<32>(block);
    while (__any_sync(FULL_MASK, len >= 32)) {
      // vie for control of warp
      if (len >= 32) {
        smem->comm[warp_id][0] = lane_id;
      }

      // winner describes adjlist
      if (smem->comm[warp_id][0] == lane_id) {
        smem->comm[warp_id][1] = left;
        smem->comm[warp_id][2] = left + len / 32 * 32;
        left += len - len % 32;
        len %= 32;
      }

      SIZE_TYPE gather = smem->comm[warp_id][1] + lane_id;
      SIZE_TYPE gather_end = smem->comm[warp_id][2];
      while (__any_sync(FULL_MASK, gather < gather_end)) {
        SIZE_TYPE mylanekey = SIZE_NONE;
        if (gather < gather_end) {
          //                    printf("thread: %d got itv32 nb: %d of node:
          //                    %d\n",thread_id,gather,node);
          mylanekey = labels[gather];
        }
        auto lmask = __match_any_sync(FULL_MASK, mylanekey);
        auto lleader = __ffs(lmask) - 1;
        auto count = __popc(lmask);
        auto hashkey = hash(mylanekey);
        hashkey = (hashkey) % htW; // 哈希表长度
        // leader update
        if (g.thread_rank() == lleader && mylanekey != SIZE_NONE) {
          auto old = smem->keys[warp_id][hashkey];
          if (old == 0 || old == mylanekey) {
            old = atomicCAS(&smem->keys[warp_id][hashkey], SIZE_TYPE(0),
                            mylanekey);
            int tempcount = atomicAdd(&smem->vals[warp_id][hashkey], count);
            count += tempcount;
            if (my_max_count < count) {
              my_max_count = count;
              my_max_key = mylanekey;
            }
          }
        }
        gather += 32;
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
    while (warp_progress < total) {
      //
      remain = total - warp_progress;

      while ((rsv_rank < warp_progress + 32) && count < len) //?
      {
        smem->left[warp_id * 32 + rsv_rank - warp_progress] = left + count++;
        rsv_rank++;
      }
      SIZE_TYPE neighbour = SIZE_NONE;
      SIZE_TYPE mylanekey = SIZE_NONE;
      if (lane_id < min(remain, 32)) {
        neighbour = smem->left[thread_id];
        mylanekey = labels[neighbour];
      }
      auto lmask = __match_any_sync(FULL_MASK, mylanekey);
      auto lleader = __ffs(lmask) - 1;
      auto count = __popc(lmask);
      auto hashkey = hash(mylanekey);
      hashkey = (hashkey) % htW; // 哈希表长度
      if (g.thread_rank() == lleader && mylanekey != SIZE_NONE) {
        auto old = smem->keys[warp_id][hashkey];
        if (old == 0 || old == mylanekey) {
          old =
              atomicCAS(&smem->keys[warp_id][hashkey], SIZE_TYPE(0), mylanekey);
          int tempcount = atomicAdd(&smem->vals[warp_id][hashkey], count);
          count += tempcount;
          if (my_max_count < count) {
            my_max_count = count;
            my_max_key = mylanekey;
          }
        }
      }
      warp_progress += 32;
    }
  }

  __device__ void medium_vertex_handle_residual_segs(CgrReader &reader,
                                                     int &my_max_key,
                                                     int &my_max_count) {
    SIZE_TYPE res_segment_cnt = reader.decode_segment_cnt();
    // each thread handle one segment
    int warp_progress = 0;
    int seg_idx = lane_id;
    while (warp_progress < res_segment_cnt) {
      volatile SIZE_TYPE offset = reader.global_offset + seg_idx * 256;
      SIZE_TYPE v_node = node;
      if (seg_idx >= res_segment_cnt) {
        v_node = SIZE_NONE;
      }

      medium_vertex_handle_one_residual_seg(v_node, offset, my_max_key,
                                            my_max_count);
      // sync
      __syncthreads();
      seg_idx += 32;
      warp_progress += 32;
    }
  }

  __device__ void
  medium_vertex_handle_one_residual_seg(SIZE_TYPE v_node,
                                        volatile SIZE_TYPE &global_offset,
                                        int &my_max_key, int &my_max_count) {
    CgrReader reader;
    reader.init(v_node, graph, global_offset);
    ResidualSegmentHelper sh(v_node, reader);
    sh.decode_residual_cnt();
    SIZE_TYPE thread_data = sh.residual_cnt;
    SIZE_TYPE rsv_rank;
    SIZE_TYPE total;
    SIZE_TYPE remain;
    auto block = cg::this_thread_block();
    auto g = cg::tiled_partition<32>(block);
    WarpScan(smem->temp_storage[warp_id])
        .ExclusiveSum(thread_data, rsv_rank, total);
    SIZE_TYPE warp_progress = 0;
    while (warp_progress < total) {
      remain = total - warp_progress;
      while ((rsv_rank < warp_progress + 32) && (sh.residual_cnt)) {
        smem->left[warp_id * 32 + rsv_rank - warp_progress] = sh.get_residual();
        rsv_rank++;
      }
      SIZE_TYPE neighbour = SIZE_NONE;
      SIZE_TYPE mylanekey = SIZE_NONE;
      if (lane_id < min(remain, 32)) {
        neighbour = smem->left[thread_id];
        // if (neighbour < 0 || neighbour > 12150976)
        // {
        //     printf("thread: %d decode illegal nb: %d of node:%d\n",
        //     thread_id, neighbour, node);
        // }
        neighbour %= N;
        mylanekey = labels[neighbour];
      }
      auto lmask = __match_any_sync(FULL_MASK, mylanekey);
      auto lleader = __ffs(lmask) - 1;
      auto count = __popc(lmask);
      auto hashkey = hash(mylanekey);
      hashkey = (hashkey) % htW;
      if (g.thread_rank() == lleader && mylanekey != SIZE_NONE) {
        auto old = smem->keys[warp_id][hashkey];
        if (old == 0 || old == mylanekey) {
          old =
              atomicCAS(&smem->keys[warp_id][hashkey], SIZE_TYPE(0), mylanekey);
          int tempcount =
              atomicAdd(&smem->vals[warp_id][hashkey], SIZE_TYPE(count));
          count += tempcount;
          if (my_max_count < count) {
            my_max_count = count;
            my_max_key = mylanekey;
          }
        }
      }
      warp_progress += 32;
    }
  }
  //
  // thread decode sequentially then propagate,
  __device__ void medium_vertex_decompress_and_lp_basic() {
    int my_max_key = -1;
    int my_max_count = 0;
    CgrReader reader;
    SIZE_TYPE row_begin = offsets[offset_index];
    reader.init(node, graph, row_begin);

    medium_vertex_handle_interval_segs_basic(reader, my_max_key, my_max_count);
    medium_vertex_handle_residual_segs(reader, my_max_key, my_max_count);
    __syncthreads();
    // warp reduce
    __syncwarp(FULL_MASK);
    typedef cub::WarpReduce<unsigned long long> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[THREAD_NUM / 32];
    unsigned long long valkey;
    valkey = my_max_key == -1
                 ? 0
                 : ((unsigned long long)my_max_count << 32) + my_max_key;
    unsigned long long aggregate =
        WarpReduce(temp_storage[warp_id]).Reduce(valkey, cub::Max());
    auto block = cg::this_thread_block();
    auto g = cg::tiled_partition<32>(block);
    if (g.thread_rank() == 0) {
      my_max_key = aggregate & 0xffffffff;
      labels[node] = my_max_key;
      // printf("thread: %d update node: %d to label: %d\n", thread_id, node,
      // my_max_key);
    }
  }

  __device__ void medium_vertex_handle_interval_segs_basic(CgrReader &reader,
                                                           int &my_max_key,
                                                           int &my_max_count) {
    SIZE_TYPE itv_segment_cnt = reader.decode_segment_cnt();
    if (itv_segment_cnt == 0) {
      return;
    }
    // each thread handle one segment
    int warp_progress = 0;
    int seg_idx = lane_id;
    while (warp_progress < itv_segment_cnt) {
      volatile SIZE_TYPE offset = reader.global_offset + seg_idx * 256;
      SIZE_TYPE v_node = node;
      if (seg_idx >= itv_segment_cnt) {
        v_node = SIZE_NONE;
      }
      handle_one_interval_seg_basic(v_node, offset, my_max_key, my_max_count);
      // sync
      if (seg_idx == itv_segment_cnt - 1) {
        smem->comm[warp_id][0] = offset;
      }
      seg_idx += 32;
      warp_progress += 32;
      // 这里需要这个同步吗？ warp级同步即可？
      __syncthreads();
    }
    // compute the write offset to decompress res nbs
    reader.global_offset = smem->comm[warp_id][0];
  }

  // baseline method handle intervals
  __device__ void
  handle_one_interval_seg_basic(SIZE_TYPE v_node,
                                volatile SIZE_TYPE &global_offset,
                                int &my_max_key, int &my_max_count) {
    CgrReader reader;
    reader.init(v_node, graph, global_offset);
    IntervalSegmentHelper sh(v_node, reader);
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
    while (warp_progress < total) {
      int remain = total - warp_progress;
      while ((rsv_rank < warp_progress + 32) && (sh.interval_cnt)) {
        smem->left[warp_id * 32 + rsv_rank - warp_progress] =
            sh.get_interval_left();
        smem->len[warp_id * 32 + rsv_rank - warp_progress] =
            sh.get_interval_len();
        rsv_rank++;
      }
      SIZE_TYPE left = 0, len = 0;
      if (lane_id < min(remain, 32)) {
        left = smem->left[thread_id];
        len = smem->len[thread_id];
      }
      expand_m_interval_basic(left, len, my_max_key, my_max_count);
      warp_progress += 32;
      __syncthreads();
    }
    // recompute global offset
    global_offset = reader.global_offset;
  }

  // advance method handle intervals

  __device__ void medium_vertex_decompress_and_lp_opt() {
    int my_max_key = -1;
    int my_max_count = 0;
    CgrReader reader;
    SIZE_TYPE row_begin = offsets[offset_index];
    reader.init(node, graph, row_begin);

    medium_vertex_handle_interval_segs_opt(reader, my_max_key, my_max_count);
    medium_vertex_handle_residual_segs(reader, my_max_key, my_max_count);
    __syncthreads();
    // warp reduce
    __syncwarp(FULL_MASK);
    typedef cub::WarpReduce<unsigned long long> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[THREAD_NUM / 32];
    unsigned long long valkey;
    valkey = my_max_key == -1
                 ? 0
                 : ((unsigned long long)my_max_count << 32) + my_max_key;
    unsigned long long aggregate =
        WarpReduce(temp_storage[warp_id]).Reduce(valkey, cub::Max());
    auto block = cg::this_thread_block();
    auto g = cg::tiled_partition<32>(block);
    if (g.thread_rank() == 0) {
      my_max_key = aggregate & 0xffffffff;
      labels[node] = my_max_key;
      // printf("thread: %d update node: %d to label: %d\n", thread_id, node,
      // my_max_key);
    }
  }

  __device__ void medium_vertex_handle_interval_segs_opt(CgrReader &reader,
                                                         int &my_max_key,
                                                         int &my_max_count) {
    SIZE_TYPE itv_segment_cnt = reader.decode_segment_cnt();
    if (itv_segment_cnt == 0) {
      return;
    }
    // each thread handle one segment
    int warp_progress = 0;
    int seg_idx = lane_id;
    while (warp_progress < itv_segment_cnt) {
      volatile SIZE_TYPE offset = reader.global_offset + seg_idx * 256;
      SIZE_TYPE v_node = node;
      if (seg_idx >= itv_segment_cnt) {
        v_node = SIZE_NONE;
      }
      handle_one_interval_seg_basic(v_node, offset, my_max_key, my_max_count);
      // sync
      if (seg_idx == itv_segment_cnt - 1) {
        smem->comm[warp_id][0] = offset;
      }
      seg_idx += 32;
      warp_progress += 32;
      // 这里需要这个同步吗？ warp级同步即可？
      __syncthreads();
    }
    // compute the write offset to decompress res nbs
    reader.global_offset = smem->comm[warp_id][0];
  }

  __device__ void handle_one_interval_seg_opt(SIZE_TYPE v_node,
                                              volatile SIZE_TYPE &global_offset,
                                              int &my_max_key,
                                              int &my_max_count) {
    CgrReader reader;
    reader.init(v_node, graph, global_offset);
    IntervalSegmentHelper sh(v_node, reader);
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
    while (warp_progress < total) {
      int remain = total - warp_progress;
      while ((rsv_rank < warp_progress + 32) && (sh.interval_cnt)) {
        smem->left[warp_id * 32 + rsv_rank - warp_progress] =
            sh.get_interval_left();
        smem->len[warp_id * 32 + rsv_rank - warp_progress] =
            sh.get_interval_len();
        rsv_rank++;
      }
      SIZE_TYPE left = 0, len = 0;
      if (lane_id < min(remain, 32)) {
        left = smem->left[thread_id];
        len = smem->len[thread_id];
      }
      expand_m_interval_opt(left, len, my_max_key, my_max_count);
      warp_progress += 32;
      __syncthreads();
    }
    // recompute global offset
    global_offset = reader.global_offset;
  }

  // write intervals directly
  __device__ void expand_m_interval_basic(SIZE_TYPE left, SIZE_TYPE len,
                                          int &my_max_key, int &my_max_count) {
    auto block = cg::this_thread_block();
    auto g = cg::tiled_partition<32>(block);
    for (SIZE_TYPE u = left; u < left + len; u++) {
      SIZE_TYPE mylanekey = labels[u];
      auto hashkey = hash(mylanekey);
      hashkey = (hashkey) % htW; // 哈希表长度
      SIZE_TYPE count = 1;
      // leader update
      if (g.thread_rank() == 0) {
        auto old = smem->keys[warp_id][hashkey];
        if (old == 0 || old == mylanekey) {
          old =
              atomicCAS(&smem->keys[warp_id][hashkey], SIZE_TYPE(0), mylanekey);
          int tempcount = atomicAdd(&smem->vals[warp_id][hashkey], count);
          count += tempcount;
          if (my_max_count < count) {
            my_max_count = count;
            my_max_key = mylanekey;
          }
        }
      }   
    }
  }

  __device__ void expand_m_interval_opt(SIZE_TYPE left, SIZE_TYPE len,
                                        int &my_max_key, int &my_max_count) {
    auto block = cg::this_thread_block();
    auto g = cg::tiled_partition<32>(block);
    while (__any_sync(FULL_MASK, len >= 32)) {
      // vie for control of warp
      if (len >= 32) {
        smem->comm[warp_id][0] = lane_id;
      }

      // winner describes adjlist
      if (smem->comm[warp_id][0] == lane_id) {
        smem->comm[warp_id][1] = left;
        smem->comm[warp_id][2] = left + len / 32 * 32;
        left += len - len % 32;
        len %= 32;
      }

      SIZE_TYPE gather = smem->comm[warp_id][1] + lane_id;
      SIZE_TYPE gather_end = smem->comm[warp_id][2];
      while (__any_sync(FULL_MASK, gather < gather_end)) {
        SIZE_TYPE mylanekey = SIZE_NONE;
        if (gather < gather_end) {
          //                    printf("thread: %d got itv32 nb: %d of node:
          //                    %d\n",thread_id,gather,node);
          mylanekey = labels[gather];
        }
        auto lmask = __match_any_sync(FULL_MASK, mylanekey);
        auto lleader = __ffs(lmask) - 1;
        auto count = __popc(lmask);
        auto hashkey = hash(mylanekey);
        hashkey = (hashkey) % htW; // 哈希表长度
        // leader update
        if (g.thread_rank() == lleader && mylanekey != SIZE_NONE) {
          auto old = smem->keys[warp_id][hashkey];
          if (old == 0 || old == mylanekey) {
            old = atomicCAS(&smem->keys[warp_id][hashkey], SIZE_TYPE(0),
                            mylanekey);
            int tempcount = atomicAdd(&smem->vals[warp_id][hashkey], count);
            count += tempcount;
            if (my_max_count < count) {
              my_max_count = count;
              my_max_key = mylanekey;
            }
          }
        }
        gather += 32;
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
    while (warp_progress < total) {
      //
      remain = total - warp_progress;

      while ((rsv_rank < warp_progress + 32) && count < len) //?
      {
        smem->left[warp_id * 32 + rsv_rank - warp_progress] = left + count++;
        rsv_rank++;
      }
      SIZE_TYPE neighbour = SIZE_NONE;
      SIZE_TYPE mylanekey = SIZE_NONE;
      if (lane_id < min(remain, 32)) {
        neighbour = smem->left[thread_id];
        mylanekey = labels[neighbour];
      }
      auto lmask = __match_any_sync(FULL_MASK, mylanekey);
      auto lleader = __ffs(lmask) - 1;
      auto count = __popc(lmask);
      auto hashkey = hash(mylanekey);
      hashkey = (hashkey) % htW; // 哈希表长度
      if (g.thread_rank() == lleader && mylanekey != SIZE_NONE) {
        auto old = smem->keys[warp_id][hashkey];
        if (old == 0 || old == mylanekey) {
          old =
              atomicCAS(&smem->keys[warp_id][hashkey], SIZE_TYPE(0), mylanekey);
          int tempcount = atomicAdd(&smem->vals[warp_id][hashkey], count);
          count += tempcount;
          if (my_max_count < count) {
            my_max_count = count;
            my_max_key = mylanekey;
          }
        }
      }
      warp_progress += 32;
    }
  }
};
