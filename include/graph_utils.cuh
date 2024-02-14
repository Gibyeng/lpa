#ifndef LPA_GRAPH_UTILS_H
#define LPA_GRAPH_UTILS_H

#include "cuda_runtime.h"
#include "../3rd_party/cub/cub/cub.cuh"
#include "../3rd_party/cub/cub/util_allocator.cuh"
#include <cuda.h>

#define FULL_MASK 0xffffffff
#define ZETA_K 3

using SIZE_TYPE = uint32_t;
using OFFSET_TYPE = uint32_t;
const int GRAPH_BYTE = 4;
const int GRAPH_LEN = 32;
static const SIZE_TYPE SIZE_NONE = 0xffffffff;
const SIZE_TYPE INTERVAL_SEGMENT_LEN = 256;

class CgrReader
{
public:
    SIZE_TYPE global_offset;
    SIZE_TYPE *graph;
    SIZE_TYPE node;

    __device__ void init(SIZE_TYPE _node, SIZE_TYPE *_graph,
                         SIZE_TYPE _global_offset)
    {
        node = _node;
        graph = _graph;
        global_offset = _global_offset;
    }

    static __device__ SIZE_TYPE decode_first_num(SIZE_TYPE node, SIZE_TYPE x)
    {
        return (x & 1) ? node - (x >> 1) - 1 : node + (x >> 1);
    }

    __device__ SIZE_TYPE cur()
    {
        OFFSET_TYPE chunk = global_offset / GRAPH_LEN;
        SIZE_TYPE buf_hi = graph[chunk];
        SIZE_TYPE buf_lo = graph[chunk + 1];
        SIZE_TYPE offset = global_offset % GRAPH_LEN;
        return __funnelshift_l(buf_lo, buf_hi, offset);
    }

    __device__ SIZE_TYPE decode_unary()
    {
        SIZE_TYPE tmp = cur();
        SIZE_TYPE x = __clz(tmp);
        global_offset += x;
        return x + 1;
    }

    __device__ SIZE_TYPE decode_int(SIZE_TYPE len)
    {
        SIZE_TYPE x = cur() >> (32 - len);
        global_offset += len;
        return x;
    }

    __device__ SIZE_TYPE decode_gamma()
    {
        SIZE_TYPE h = decode_unary();
        return this->decode_int(h) - 1;
    }

#if ZETA_K != 1
    __device__ SIZE_TYPE decode_zeta()
    {
        SIZE_TYPE h = decode_unary();
        global_offset++;
        SIZE_TYPE x = decode_int(h * ZETA_K);
        return x - 1;
    }
#endif

    __device__ SIZE_TYPE decode_residual_code()
    {
#if ZETA_K == 1
        return decode_gamma();
#else
        return decode_zeta();
#endif
    }

    __device__ SIZE_TYPE decode_segment_cnt()
    {
        SIZE_TYPE segment_cnt = node == SIZE_NONE ? 0 : decode_gamma() + 1;
        if (segment_cnt == 1 && (cur() & 0x80000000))
        {
            global_offset += 1;
            segment_cnt = 0;
        }
        return segment_cnt;
    }
};

struct ResidualSegmentHelper
{
    SIZE_TYPE residual_cnt;

    SIZE_TYPE left;
    bool first_res;

    CgrReader &cgrr;

    __device__ ResidualSegmentHelper(SIZE_TYPE node, CgrReader &cgrr)
        : cgrr(cgrr), first_res(true), left(0), residual_cnt(0) {}

    __device__ void decode_residual_cnt()
    {
        residual_cnt = cgrr.node == SIZE_NONE ? 0 : cgrr.decode_gamma();
    }

    __device__ SIZE_TYPE get_residual()
    {
        if (first_res)
        {
            left = decode_first_num();
            first_res = false;
        }
        else
        {
            left += cgrr.decode_residual_code() + 1;
        }
        residual_cnt--;
        return left;
    }

    __device__ SIZE_TYPE decode_first_num()
    {
        SIZE_TYPE x = cgrr.decode_residual_code();
        return (x & 1) ? cgrr.node - (x >> 1) - 1 : cgrr.node + (x >> 1);
    }
};

struct IntervalSegmentHelper
{
    SIZE_TYPE interval_cnt;

    SIZE_TYPE left;
    bool first_interval;

    CgrReader &cgrr;

    __device__ IntervalSegmentHelper(SIZE_TYPE node, CgrReader &cgrr)
        : cgrr(cgrr), first_interval(true), left(0), interval_cnt(0) {}

    __device__ void decode_interval_cnt()
    {
        interval_cnt = cgrr.node == SIZE_NONE ? 0 : cgrr.decode_gamma();
    }

    __device__ SIZE_TYPE get_interval_left()
    {
        if (first_interval)
        {
            left = decode_first_num();
            first_interval = false;
        }
        else
        {
            left += cgrr.decode_gamma() + 1;
        }
        return left;
    }

    __device__ SIZE_TYPE get_interval_len()
    {
        SIZE_TYPE len = cgrr.decode_gamma() + 4;
        left += len;
        interval_cnt--;
        return len;
    }

    __device__ SIZE_TYPE decode_first_num()
    {
        SIZE_TYPE x = cgrr.decode_gamma();
        return (x & 1) ? cgrr.node - (x >> 1) - 1 : cgrr.node + (x >> 1);
    }
};

#endif // LPA_GRAPH_UTILS_H
