#pragma once

#include <stdio.h>
#include <iostream>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "decoder.cuh"
#include "file.h"
#include "compressor.h"

const int THREADS_NUM = 128;

__global__ void propagate_large_vertex(int end,
									   SIZE_TYPE *graph,
									   SIZE_TYPE *vertices,
									   SIZE_TYPE *offsets,
									   SIZE_TYPE *row,
									   int *labels,
									   int N
									   )
{
	int block_id = blockIdx.x;
	int steps = gridDim.x;
	// 默认大点从下标0 开始
	for (int i = block_id; i < end; i += steps)
	{
		__shared__ typename Decoder<4, 100, 128>::SMem_b smem;
		SIZE_TYPE v = vertices[i];
		Decoder<4, 100, 128> decoder(&smem, graph, offsets, row, labels, v, i,N);
		decoder.decompress_and_lp();
	}
}

__global__ void propagate_mid_vertex(int begin,
									 int end,
									 SIZE_TYPE *graph,
									 SIZE_TYPE *vertices,
									 SIZE_TYPE *offset,
									 SIZE_TYPE *row,
									 int *labels,
									 int N
									 )
{
	int tid_in_all = threadIdx.x + blockIdx.x * blockDim.x;
	int steps = gridDim.x * (blockDim.x / 32);
	int warpid_in_all = (tid_in_all) / 32;
	for (int i = warpid_in_all + begin; i < end; i += steps)
	{
		__shared__ typename Decoder<4, 100, 128>::SMem_b smem;
		SIZE_TYPE v = vertices[i];
		Decoder<4, 100, 128> decoder(&smem, graph, offset, row, labels, v, i,N);
		//decoder.decompress_and_lp();
	}
}

template <typename V>
__global__ void initialize_labels(V *labels, V n)
{
	for (auto i : grid_stride_range(n))
	{
		labels[i] = i;
	}
}

// TW = thread warp,default =32, VT=1,NT
template <int TW, int VT, int NT>
__global__ void l_small_update_syn_labelload(int offset, int n, SIZE_TYPE *neighbors, SIZE_TYPE *offsets, int *labels_write, int *labels_read, SIZE_TYPE *vertices, int *counter)
{
	int mylanekey;
	int my_max_key;

	auto block = cg::this_thread_block();
	auto g = cg::tiled_partition<TW>(block);
	// the index of warp(group)
	int warpid = threadIdx.x / TW;
// the vertex about to process
#pragma unroll
	for (int stride = 0; stride < VT; ++stride)
	{
		auto id = (threadIdx.x + VT * blockIdx.x * blockDim.x + stride * blockDim.x) / TW + offset;
		if (id < n)
		{
			const int v = id;
			//printf(" %d \n",id);
			const int begin = offsets[v];
			const int end = offsets[v + 1];
			//load value
			//no neighbor
			if (begin == end)
			{
				my_max_key = v + 1;
				labels_write[v] = my_max_key - 1;
			}
			else
			{
				// number of neighbors is small;
				int j = g.thread_rank() + begin;
				// sample first TW label
				if (j < end)
				{
					mylanekey = labels_read[neighbors[j]] + 1;
				}
				else
				{
					mylanekey = 0;
				}

				auto gmask = __match_any_sync(FULL_MASK, mylanekey);

				int count = 0;
				if (g.thread_rank() == __ffs(gmask) - 1 && mylanekey != 0)
				{
					count = __popc(gmask);
				}

				unsigned long long valkey = ((unsigned long long)count << 32) + mylanekey;

				// cub reduce
				typedef cub::WarpReduce<unsigned long long> WarpReduce;
				__shared__ typename WarpReduce::TempStorage temp_storage[NT / TW];
				unsigned long long aggregate = WarpReduce(temp_storage[warpid]).Reduce(valkey, cub::Max());
				if (g.thread_rank() == 0)
				{
					my_max_key = aggregate;
					auto lbl = labels_read[v];
					if (lbl != my_max_key - 1)
					{
						//						atomicAdd(counter, 1);
						labels_write[v] = my_max_key - 1;
						//printf("change v %d, label %d to %d \n",v ,lbl,my_max_key - 1);
					}
				}
			}
		}
	}
}

__global__ void l_tiny_update_syn2_labelload(SIZE_TYPE *neighbors, SIZE_TYPE *offsets, int *labels_write, int *labels_read, int *counter, int *warp_v, int *warp_begin, int warpnumber)
{
	auto block = cg::this_thread_block();
	auto g = cg::tiled_partition<32>(block);
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int warpIdAll = (tid) / 32;
	int v = -1;
	if (warpIdAll < warpnumber)
	{
		v = warp_v[tid];
	}
	const int begin = warp_begin[warpIdAll];
	// load nodes. group lanes by node id.
	auto mask = __ballot_sync(FULL_MASK, v != -1);
	int label;
	if (v != -1)
	{
		label = labels_read[neighbors[begin + g.thread_rank()]] + 1;
	}
	else
	{
		label = 0;
	}

	// compute head_flag.
	auto vmask = __match_any_sync(FULL_MASK, v);
	auto lmask = __match_any_sync(vmask, label);
	int count = 0;
	int head = 0;
	if (g.thread_rank() == __ffs(vmask) - 1)
	{
		head = 1;
	}
	if (g.thread_rank() == __ffs(lmask) - 1)
	{
		count = __popc(lmask);
	}

	typedef cub::WarpReduce<unsigned long long> WarpReduce;
	__shared__ typename WarpReduce::TempStorage temp_storage;
	unsigned long long valkey = ((unsigned long long)count << 32) + label;
	unsigned long long aggregate = WarpReduce(temp_storage).HeadSegmentedReduce(valkey, head, cub::Max());
	if (g.thread_rank() == __ffs(vmask) - 1 && v != -1)
	{
		label = aggregate;
		auto lbl = labels_read[v];
		if (lbl != label - 1)
		{
			//			atomicAdd(counter, 1);
			labels_write[v] = label - 1;
		}
	}
}
