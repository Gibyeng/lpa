//
// Created by tangmengqiu on 2020/5/24.
//

#pragma once

#include "../../3rd_party/cub/cub/cub.cuh"
#include "kernel_scan.hxx"
#include "kernel_segsort.hxx"
#include "memory.hxx"
#include "transform.hxx"
#include <cuda_profiler_api.h>

#include "cms.cuh"
#include "decoder.cuh"
#include "kernel.cuh"
#include "label_propagator.h"
#include "utils.h"

using std::cout;
using std::endl;

template <typename V, typename E>
class NovaPlusPropagator : public LabelPropagator<V, E> {
public:
  using typename LabelPropagator<V, E>::CGRGraphT;
  NovaPlusPropagator(std::shared_ptr<CGRGraphT> _G)
      : LabelPropagator<V, E>(_G->n), G(_G) {}

  std::pair<double, double> run(int niter);

  using LabelPropagator<V, E>::labels;

  // 保存图数据
  std::shared_ptr<CGRGraphT> G;

  void preprocess();

  int iterate(int i);

  void postprocess();

  int get_count();

  void errorCheck(std::string message);

  void transfer_data();

  void tiny_node_assignment();

  void init_gmem();

  void free_gmem();

private:
  // GPU memory
  // alloced and transfer data
  // 压缩数据
  int *d_labels;      // n
  E *d_csr_offsets;   // n
  V *d_cgr_neighbors; // m1
  E *d_cgr_offsets;   // n1
  V *d_vertices;      // n

  // 非压缩数据
  V *d_small_neighbors; // m2
  E *d_small_offsets;   // n2
  int *d_counter;       // 1

  int num_big;
  int num_medium;
  int num_small;
  int num_tiny;
  int big_index;
  int medium_index;
  int small_index;
  SIZE_TYPE medium;
  SIZE_TYPE big;
  SIZE_TYPE small;

  int b_block_num;
  int m_block_num;
  // tiny nodes
  int warpnumber;
  int *d_warp_v;
  int *d_warp_begin;
  int nt_tiny;
  int nb_tiny2;
};

template <typename V, typename E>
void NovaPlusPropagator<V, E>::errorCheck(std::string message) {
  auto err = cudaGetLastError();
  if (cudaSuccess != err) {
    printf("Error! %s : %s\n", message.c_str(), cudaGetErrorString(err));
  }
}

template <typename V, typename E>
std::pair<double, double> NovaPlusPropagator<V, E>::run(int niter) {
  Timer t1, t2;
  t2.start();
  preprocess();

  const int n = G->n;
  const int nthreads = 128;
  const int n_blocks = min(512, divup(n, nthreads));
  initialize_labels<<<n_blocks, nthreads>>>(d_labels, n);
  cudaDeviceSynchronize();
  errorCheck("after initialize labels");

  t1.start();
  // Main loop
  for (auto i = 0; i < niter; i++) {
    Timer t_iter;
    t_iter.start();
    cout << "iteration: " << i << " begin" << endl;
    int count = iterate(i);
    cout << "iteration: " << i << " done" << endl;
    t_iter.stop();
    printf("%d: %f (ms), updated %d\n", i + 1, t_iter.elapsed_time(), count);
  }
  cudaMemcpy(labels.get(), d_labels, sizeof(V) * n, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  t1.stop();
  t2.stop();
  postprocess();
  return std::make_pair(t1.elapsed_time(), t2.elapsed_time());
}

template <typename V, typename E> int NovaPlusPropagator<V, E>::iterate(int i) {
  errorCheck("before LP!");
  Timer t_all;
  t_all.start();
  // launch big node kernel
  if (num_big > 0) {
    cout << "BIG NODE LPA starts: " << endl;
    Timer t_big;
    t_big.start();
    propagate_large_vertex<<<b_block_num, 128>>>(big_index, d_cgr_neighbors,
                                                 d_vertices, d_cgr_offsets,
                                                 d_csr_offsets, d_labels, G->n);
    cudaDeviceSynchronize();
    t_big.stop();

    errorCheck("after big nodes LP!");
    cout << "compute big kernel time:" << t_big.elapsed_time() << endl;
    cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
  }

  // launch medium node kernel
  if (num_medium > 0) {
    cout << "MID NODE LPA starts: " << endl;
    Timer t_mid;
    t_mid.start();

    propagate_mid_vertex<<<m_block_num, 128>>>(
        big_index, medium_index, d_cgr_neighbors, d_vertices, d_cgr_offsets,
        d_csr_offsets, d_labels, G->n);
    cudaDeviceSynchronize();
    errorCheck("after mid nodes LP!");
    t_mid.stop();
    cout << "compute mid kernel time :" << t_mid.elapsed_time() << endl;
    cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
  }
  if (num_small > 0) {
    Timer t_small;
    t_small.start();
    const int nb_small = divup(num_small, 2);
    const int nt_small = 32 * 2;
    l_small_update_syn_labelload<32, 1, 64><<<nb_small, nt_small>>>(
        medium_index, num_small, d_small_neighbors, d_small_offsets, d_labels,
        d_labels, d_vertices, d_counter);
    cudaDeviceSynchronize();
    errorCheck("after small nodes LP!");
    t_small.stop();
    cout << "update small nodes label elapsed: " << t_small.elapsed_time()
         << endl;
    cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
  }
  if (num_tiny > 0) {

    Timer t_tiny;
    t_tiny.start();
    l_tiny_update_syn2_labelload<<<nb_tiny2, nt_tiny>>>(
        d_small_neighbors, d_small_offsets, d_labels, d_labels, d_counter,
        d_warp_v, d_warp_begin, warpnumber);
    cudaDeviceSynchronize();
    errorCheck("after tinys nodes LP!");
    t_tiny.stop();
    cout << "update tiny nodes label elapsed: " << t_tiny.elapsed_time()
         << endl;
    cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
  }
  t_all.stop();
  cout << "nova lp elapsed: " << t_all.elapsed_time() << endl;
  cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
  return get_count();
}

template <typename V, typename E> void NovaPlusPropagator<V, E>::preprocess() {
  tiny_node_assignment();
  init_gmem();
  transfer_data();
}

template <typename V, typename E>
void NovaPlusPropagator<V, E>::tiny_node_assignment() {
  num_big = 0;
  num_medium = 0;
  num_small = 0;
  num_tiny = 0;
  // record the index in vectices array.
  big_index = 0;
  medium_index = 0;
  small_index = 0;
  medium = G->small_node_thresh;
  cout << "mediun node nb > " << medium << endl;
  big = 128;
  small = 16;

  // computer nodes size
  big_index =
      binary_search(SIZE_TYPE(0), G->n - 1, G->csr_offsets, G->vertices, big);
  printf("\nbig_index %d \n", big_index);
  medium_index = medium == 0
                     ? G->n - 1
                     : binary_search(SIZE_TYPE(0), G->n - 1, G->csr_offsets,
                                     G->vertices, medium);
  printf("medium_index %d \n", medium_index);

  if (G->small_node_thresh > 16) {
    // in this case , both small nodes and tiny nodes exist
    small_index = binary_search(SIZE_TYPE(0), G->n - 1, G->csr_offsets,
                                G->vertices, small);

    num_tiny = G->n - 1 - small_index;
    cout << "small_index: " << small_index << endl;
  } else {
    // small nodes not exist, tiny nodes only
    small_index = medium_index;
  }
  num_big = big_index;
  num_medium = medium_index - big_index;
  num_small = G->small_node_thresh > 16 ? small_index - medium_index : 0;

  printf("num_big: %d,num_medium: %d,num_small: %d,num_tiny: %d\n", num_big,
         num_medium, num_small, num_tiny);

  b_block_num = (num_big < 512) ? num_big : 512;
  printf("block num for big nodes :%d\n", b_block_num);
  m_block_num = (num_medium < 512) ? divup(num_medium, 4) : 512;
  printf("block num for mid nodes :%d\n", m_block_num);
  // compute tiny nodes, how many edges for each warps
  warpnumber = 1;
  std::vector<int> warp_v;
  std::vector<int> warp_begin;

  int start = small_index;
  if (small_index == -1) {
    // all node are tiny
    start = 0;
  }
  warp_begin.push_back(start);
  for (int j = 0; j < num_tiny; j++) {
    int node = G->vertices[j + small_index];
    int node_nb_num = G->csr_offsets[node + 1] - G->csr_offsets[node];
    int start_node = G->vertices[start];
    if (G->csr_offsets[node + 1] - G->csr_offsets[start_node] > 32) {
      warpnumber++;
      int align_minus_1 =
          32 - (G->csr_offsets[node] - G->csr_offsets[start_node]);
      for (int k = 0; k < align_minus_1; k++) {
        warp_v.push_back(-1);
      }
      for (int k = 0; k < node_nb_num; k++) {
        warp_v.push_back(node);
      }
      start = small_index + j;
      warp_begin.push_back(G->csr_small_offsets[j]);
    } else {
      for (int k = 0; k < node_nb_num; k++) {
        warp_v.push_back(node);
      }
    }
  }
  int align = warpnumber * 32 - (int)warp_v.size();
  if (align != 0) {
    while (align) {
      warp_v.push_back(-1);
      align--;
    }
  }
  cout << "warpnumber: " << warpnumber << endl;
  nt_tiny = 64;
  nb_tiny2 = divup(warpnumber, nt_tiny / 32);
  cudaMalloc(&d_warp_v, sizeof(int) * (warpnumber * 32));
  cudaMalloc(&d_warp_begin, sizeof(int) * warpnumber);
  cudaMemcpy(d_warp_v, &warp_v[0], sizeof(int) * (warpnumber * 32),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_warp_begin, &warp_begin[0], sizeof(int) * warpnumber,
             cudaMemcpyHostToDevice);
  printf("tiny nodes assignment success \n");
}

template <typename V, typename E> void NovaPlusPropagator<V, E>::init_gmem() {
  {
    cout << "GPU MEM ALLOC INFO: " << endl;
    cout << "d_labels:         " << G->n << endl;
    cout << "d_csr_offsets:    " << G->csr_offsets.size() << endl;
    cout << "d_cgr_neighbors:  " << G->cgr_neighbors.size() << endl;
    cout << "d_cgr_offsets:    " << G->cgr_offsets.size() << endl;
    cout << "d_vertices:       " << G->vertices.size() << endl;
    cout << "d_small_neighbors:" << G->csr_small_neighbors.size() << endl;
    cout << "d_small_offsets:  " << G->csr_small_offsets.size() << endl;
    cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
  }
  cudaMalloc(&d_labels, sizeof(int) * G->n);

  cudaMalloc(&d_csr_offsets, sizeof(int) * G->csr_offsets.size());

  cudaMalloc(&d_cgr_neighbors, sizeof(int) * G->cgr_neighbors.size());

  cudaMalloc(&d_cgr_offsets, sizeof(int) * G->cgr_offsets.size());

  cudaMalloc(&d_vertices, sizeof(int) * G->vertices.size());

  cudaMalloc(&d_small_neighbors, sizeof(int) * G->csr_small_neighbors.size());

  cudaMalloc(&d_small_offsets, sizeof(int) * G->csr_small_offsets.size());

  cudaMalloc(&d_counter, sizeof(int) * 1);

  cudaMemset(d_counter, 0, sizeof(int));

  errorCheck("GPU mem alloc");
}

template <typename V, typename E>
void NovaPlusPropagator<V, E>::transfer_data() {
  cout << "Date transfer,host2device: " << endl;
  Timer t;
  int n = G->vertices.size();
  int m1 = G->cgr_neighbors.size();
  int n1 = G->cgr_offsets.size();
  int m2 = G->csr_small_neighbors.size();

  int n2 = G->csr_small_offsets.size();
  t.start();
  // 需要传输的数据为： n + m1 + n1 + m2+ n2+ n= 3n + m1 + m2
  cudaMemcpy(d_csr_offsets, &G->csr_offsets[0],
             sizeof(V) * G->csr_offsets.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cgr_neighbors, &G->cgr_neighbors[0], sizeof(V) * m1,
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_cgr_offsets, &G->cgr_offsets[0], sizeof(V) * n1,
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_small_neighbors, &G->csr_small_neighbors[0], sizeof(V) * m2,
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_small_offsets, &G->csr_small_offsets[0], sizeof(V) * n2,
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_vertices, &G->vertices[0], sizeof(V) * n,
             cudaMemcpyHostToDevice);

  t.stop();
  cout << "n= " << n << " m1= " << m1 << " m2= " << m2 << endl;
  cout << "transfer data: [ 3n+m1+m2= " << 3 * n + m1 + m2
       << " ] elapse: " << t.elapsed_time() << endl;
  cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
}

template <typename V, typename E> void NovaPlusPropagator<V, E>::postprocess() {
  free_gmem();
}

template <typename V, typename E> void NovaPlusPropagator<V, E>::free_gmem() {
  cudaFree(d_labels);
  cudaFree(d_csr_offsets);
  cudaFree(d_cgr_neighbors);
  cudaFree(d_cgr_offsets);
  cudaFree(d_vertices);
  cudaFree(d_small_neighbors);
  cudaFree(d_small_offsets);
  cudaFree(d_counter);

  errorCheck("GPU mem free");
  return;
}

// Return the number of labels updated
template <typename V, typename E> int NovaPlusPropagator<V, E>::get_count() {
  int counter;
  cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemset(d_counter, 0, sizeof(int));
  return counter;
}
