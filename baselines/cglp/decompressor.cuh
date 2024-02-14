//
// Created by Rich on 2021/1/25.
//

#pragma once

// cuda
#include "../../3rd_party/cub/cub/cub.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cms.cuh"
#include "graph_utils.cuh"

#include <iostream>
#include <stdio.h>

template <int w, int d, int htW, int THREADS_NUM = 128> class Decoder2 {
public:
  typedef cub::BlockScan<SIZE_TYPE, THREADS_NUM> BlockScan;
  typedef cub::WarpScan<SIZE_TYPE> WarpScan;

  struct SMem_b {
    typename BlockScan::TempStorage block_temp_storage;
    typename WarpScan::TempStorage temp_storage[THREADS_NUM / 32];
    volatile SIZE_TYPE left[THREADS_NUM];
    volatile SIZE_TYPE len[THREADS_NUM];
    volatile SIZE_TYPE comm[THREADS_NUM / 32][32];
  };
  Decoder2(){};
  __device__ Decoder2(SMem_b *_smem, SIZE_TYPE *_graph, SIZE_TYPE *_offsets,
                      SIZE_TYPE *_row, int _node, int _offset_index)
      : smem(_smem), graph(_graph), offsets(_offsets), row(_row), node(_node),
        offset_index(_offset_index) {
    thread_id = threadIdx.x;
    lane_id = thread_id % 32;
    warp_id = thread_id / 32;
    tid_in_all = threadIdx.x + blockIdx.x * blockDim.x;
  }
  __device__ void decompress() {
    if (row[node + 1] - row[node] >= THREADS_NUM) {
      large_vertex_decompress();
    } else {
      medium_vertex_decompress();
    }
    return;
  }

private:
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

  // one node one block
  __device__ void large_vertex_decompress() {
    CgrReader reader;
    SIZE_TYPE row_begin = offsets[offset_index];
    reader.init(node, graph, row_begin);
    large_vertex_handle_interval_segs(reader);
    large_vertex_handle_residual_segs(reader);
    __syncthreads();
    return;
  }

  __device__ void large_vertex_handle_interval_segs(CgrReader &reader) {
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
      large_vertex_handle_one_interval_seg(v_node, offset);

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
                                       volatile SIZE_TYPE &global_offset) {
    CgrReader reader;
    reader.init(v_node, graph, global_offset);
    IntervalSegmentHelper sh(v_node, reader);
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
      expand_interval(left, len);
      cta_progress += THREADS_NUM;
      __syncthreads();
    }
    // compute the global offset
    global_offset = reader.global_offset;
  }

  __device__ void expand_interval(SIZE_TYPE left, SIZE_TYPE len) {
    expand_itv_more_than_128(left, len);
    expand_itv_more_than_32(left, len);
    expand_itv_more_than_0(left, len);
    return;
  }

  __device__ void expand_itv_more_than_128(SIZE_TYPE &left, SIZE_TYPE &len) {
    // 如果某个len长度大于thread num
    // 则block中全部的thread都去处理这个 <left,len>
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
        //
        gather += THREADS_NUM;
      }
    }
  }

  __device__ void expand_itv_more_than_32(SIZE_TYPE &left, SIZE_TYPE &len) {
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
        if (gather < gather_end) {
          // mock
        }

        gather += 32;
      }
    }
  }

  __device__ void expand_itv_more_than_0(SIZE_TYPE &left, SIZE_TYPE &len) {
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
    while (cta_progress < total) {
      remain = total - cta_progress;
      while ((rsv_rank < cta_progress + THREADS_NUM) && (cnt < len)) {
        smem->left[rsv_rank - cta_progress] = left++;
        rsv_rank++;
        cnt++;
      }
      __syncthreads();
      SIZE_TYPE neighbour = SIZE_NONE;
      // int mylanekey = -1;
      if (thread_id < min(remain, THREADS_NUM)) {
        neighbour = smem->left[thread_id];
        // mock
      }
      cta_progress += THREADS_NUM;
      __syncthreads();
    }
  }

  __device__ void large_vertex_handle_residual_segs(CgrReader &reader) {

    SIZE_TYPE res_segment_cnt = reader.decode_segment_cnt();
    int block_progress = 0;
    int seg_idx = thread_id; // 没必要 warp - res_seg
    while (block_progress < res_segment_cnt) {
      volatile SIZE_TYPE offset = reader.global_offset + seg_idx * 256;
      SIZE_TYPE v_node = node;
      if (seg_idx >= res_segment_cnt) {
        v_node = SIZE_NONE;
      }
      large_vertex_handle_one_residual_seg(v_node, offset);
      // sync
      __syncthreads();
      seg_idx += THREADS_NUM;
      block_progress += THREADS_NUM;
    }
  }
  // one thread one segment
  __device__ void
  large_vertex_handle_one_residual_seg(SIZE_TYPE v_node,
                                       volatile SIZE_TYPE &global_offset) {
    CgrReader reader;
    reader.init(v_node, graph, global_offset);
    ResidualSegmentHelper sh(v_node, reader);
    sh.decode_residual_cnt();
    SIZE_TYPE thread_data = sh.residual_cnt;
    SIZE_TYPE total;
    SIZE_TYPE remain;
    SIZE_TYPE rsv_rank;
    __syncthreads();
    BlockScan(smem->block_temp_storage)
        .ExclusiveSum(thread_data, rsv_rank, total);
    __syncthreads();
    SIZE_TYPE cta_progress = 0;

    while (cta_progress < total) {
      remain = total - cta_progress;
      while (rsv_rank < cta_progress + THREADS_NUM && sh.residual_cnt) {
        SIZE_TYPE neighbour = sh.get_residual();
        smem->left[rsv_rank - cta_progress] = neighbour;
        rsv_rank++;
      }
      __syncthreads();
      SIZE_TYPE neighbour = SIZE_NONE;
      int mylanekey = -1;
      if (thread_id < min(remain, THREADS_NUM)) {
        // mock
        neighbour = smem->left[thread_id];
      }
      cta_progress += THREADS_NUM;
      __syncthreads();
    }
  }

  __device__ void medium_vertex_decompress() {
    int my_max_key = -1;
    int my_max_count = 0;
    CgrReader reader;
    SIZE_TYPE row_begin = offsets[offset_index];
    reader.init(node, graph, row_begin);

    medium_vertex_handle_interval_segs(reader);
    medium_vertex_handle_residual_segs(reader);
  }

  __device__ void medium_vertex_handle_interval_segs(CgrReader &reader) {
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
      medium_vertex_handle_one_interval_seg(v_node, offset);
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
                                        volatile SIZE_TYPE &global_offset) {
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
      expand_m_interval(left, len);
      warp_progress += 32;
      __syncthreads();
    }
    // recompute global offset
    global_offset = reader.global_offset;
  }

  __device__ void expand_m_interval(SIZE_TYPE left, SIZE_TYPE len) {

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
          // mock
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
      if (lane_id < min(remain, 32)) {
        neighbour = smem->left[thread_id];
      }
      warp_progress += 32;
    }
  }

  __device__ void medium_vertex_handle_residual_segs(CgrReader &reader) {
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

      medium_vertex_handle_one_residual_seg(v_node, offset);
      // sync
      __syncthreads();
      seg_idx += 32;
      warp_progress += 32;
    }
  }

  __device__ void
  medium_vertex_handle_one_residual_seg(SIZE_TYPE v_node,
                                        volatile SIZE_TYPE &global_offset) {
    CgrReader reader;
    reader.init(v_node, graph, global_offset);
    ResidualSegmentHelper sh(v_node, reader);
    sh.decode_residual_cnt();
    SIZE_TYPE thread_data = sh.residual_cnt;
    SIZE_TYPE rsv_rank;
    SIZE_TYPE total;
    SIZE_TYPE remain;
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
      }
      warp_progress += 32;
    }
  }
};
