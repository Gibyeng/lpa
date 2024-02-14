#pragma once
#include "range.cuh"
#include <memory>
# include <cmath>
# include <limits>
#define LONG_PRIME 4294967311
#define CMS_D 5
template<int W,int d>
class SharedCM {
public:
    uint32_t h1vals[W];
#if CMS_D>1
    uint32_t h2vals[W];
#endif
#if CMS_D>2
    uint32_t h3vals[W];
#endif
#if CMS_D>3
    uint32_t h4vals[W];
#endif
#if CMS_D>4
    uint32_t h5vals[W];
#endif


    __device__ uint32_t increment(uint32_t key, uint32_t x=1) {
        // auto c = _increment( h1vals, 0, key, x);
        int c;
#if CMS_D == 1
        c = _increment_1( h1vals, 0, key, x);
#endif
#if CMS_D==2
        c = _increment_2( h1vals,h2vals, 0, key, x);
#endif
#if CMS_D==3
        c = _increment_3( h1vals,h2vals,h3vals, 0, key, x);
#endif
#if CMS_D==4
        c = _increment_4( h1vals,h2vals,h3vals, h4vals,0, key, x);
#endif
#if CMS_D==5
        c = _increment_5( h1vals,h2vals,h3vals,h4vals,h5vals, 0, key, x);
#endif

        return c;
    }

    __device__ void clear() {


        for (auto i: block_stride_range(W)) {

            h1vals[i] = 0;
#if CMS_D>1
            h2vals[i] = 0;
#endif
#if CMS_D>2
            h3vals[i] = 0;
#endif
#if CMS_D>3
            h4vals[i] = 0;
#endif
#if CMS_D>4
            h5vals[i] = 0;
#endif
        }
    }

//    __device__ uint32_t cm_hash1(uint32_t k) {
//
//        // calculate a in hash function
//        uint32_t a = 12121;
//        // calculate b in hash function
//        uint32_t b = 131;
//        return ((uint64_t)a*k+b)%LONG_PRIME%INT32_MAX;
//    }
//
//    __device__ uint32_t cm_hash2(uint32_t k) {
//
//        // calculate a in hash function
//        uint32_t a = 5324245;
//        // calculate b in hash function
//        uint32_t b = 2324;
//        return ((uint64_t)a*k+b)%LONG_PRIME%INT32_MAX;
//    }
//
//    __device__ uint32_t cm_hash3(uint32_t k) {
//
//        // calculate a in hash function
//        uint32_t a = 22122;
//        // calculate b in hash function
//        uint32_t b = 235;
//        return ((uint64_t)a*k+b)%LONG_PRIME%INT32_MAX;
//    }
//
    __device__ uint32_t cm_hash1(uint32_t k) {

        k ^= k >> 16;
        k *= 0x12ebcb3b;
        k ^= k >> 13;
        k *= 0xc123ae53;
        k ^= k >> 16;
        return k;
    }

    __device__ uint32_t cm_hash2(uint32_t k) {

        k ^= k >> 16;
        k *= 0x21eb21bb;
        k ^= k >> 13;
        k *= 0x2cb2ea35;
        k ^= k >> 16;
        return k;
    }

    __device__ uint32_t cm_hash3(uint32_t k) {

        k ^= k >> 16;
        k *= 0x8febce61;
        k ^= k >> 13;
        k *= 0xa2b2be25;
        k ^= k >> 16;
        return k;
    }
    __device__ uint32_t cm_hash4(uint32_t k) {
        k ^= k >> 12;
        k *= 0x13ebdb3a;
        k ^= k >> 12;
        k *= 0xc123ae53;
        k ^= k >> 14;
        return k;
    }

    __device__ uint32_t cm_hash5(uint32_t k) {
        k ^= k >> 13;
        k *= 0x312b521b;
        k ^= k >> 5;
        k *= 0x1cb3ea22;
        k ^= k >> 17;
        return k;
    }


    __device__ uint32_t _increment_1
            ( uint32_t *h1vals, uint32_t begin, uint32_t key, uint32_t x=1)
    {

        uint32_t i_1 = cm_hash1(key)%W;

        auto v_1 = atomicAdd(&h1vals[begin + i_1], x);

        return v_1 + x;

    }

    __device__ uint32_t _increment_2
            ( uint32_t *h1vals,uint32_t *h2vals, uint32_t begin, uint32_t key, uint32_t x=1)
    {

        uint32_t i_1 = cm_hash1(key)%W;

        uint32_t i_2 = cm_hash2(key)%W;

        auto v_1 = atomicAdd(&h1vals[begin + i_1], x);

        auto v_2 = atomicAdd(&h2vals[begin + i_2], x);

        v_1 = min(v_1,v_2);

        return v_1 + x;

    }

    __device__ uint32_t _increment_3
            ( uint32_t *h1vals,uint32_t *h2vals,uint32_t *h3vals, uint32_t begin, uint32_t key, uint32_t x=1)
    {

        uint32_t i_1 = cm_hash1(key)%W;

        uint32_t i_2 = cm_hash2(key)%W;

        uint32_t i_3 = cm_hash3(key)%W;

        auto v_1 = atomicAdd(&h1vals[begin + i_1], x);

        auto v_2 = atomicAdd(&h2vals[begin + i_2], x);
        v_1 = min(v_1,v_2);

        auto v_3 = atomicAdd(&h2vals[begin + i_3], x);
        v_1 = min(v_1,v_3);

        return v_1 + x;

    }

    __device__ uint32_t _increment_4
            ( uint32_t *h1vals,uint32_t *h2vals,uint32_t *h3vals,uint32_t *h4vals, uint32_t begin, uint32_t key, uint32_t x=1)
    {

        uint32_t i_1 = cm_hash1(key)%W;

        uint32_t i_2 = cm_hash2(key)%W;

        uint32_t i_3 = cm_hash3(key)%W;

        uint32_t i_4 = cm_hash4(key)%W;

        auto v_1 = atomicAdd(&h1vals[begin + i_1], x);

        auto v_2 = atomicAdd(&h2vals[begin + i_2], x);
        v_1 = min(v_1,v_2);

        auto v_3 = atomicAdd(&h2vals[begin + i_3], x);
        v_1 = min(v_1,v_3);

        auto v_4 = atomicAdd(&h2vals[begin + i_3], x);
        v_1 = min(v_1,v_4);

        return v_1 + x;

    }

    __device__ uint32_t _increment_5
            ( uint32_t *h1vals,uint32_t *h2vals,uint32_t *h3vals,uint32_t *h4vals ,uint32_t *h5vals,uint32_t begin, uint32_t key, uint32_t x=1)
    {

        uint32_t i_1 = cm_hash1(key)%W;

        uint32_t i_2 = cm_hash2(key)%W;

        uint32_t i_3 = cm_hash3(key)%W;

        uint32_t i_4 = cm_hash4(key)%W;

        uint32_t i_5 = cm_hash5(key)%W;



        auto v_1 = atomicAdd(&h1vals[begin + i_1], x);

        auto v_2 = atomicAdd(&h2vals[begin + i_2], x);
        v_1 = min(v_1,v_2);


        auto v_3 = atomicAdd(&h3vals[begin + i_3], x);
        v_1 = min(v_1,v_3);


        auto v_4 = atomicAdd(&h4vals[begin + i_4], x);
        v_1 = min(v_1,v_4);


        auto v_5 = atomicAdd(&h5vals[begin + i_5], x);
        v_1 = min(v_1,v_5);


        return v_1 + x;

    }

};