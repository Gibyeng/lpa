// -*- coding: utf-8 -*-

#pragma once

// cub
#include "../../3rd_party/cub/cub/cub.cuh"
// mgpu
#include "kernel_scan.hxx"
#include "kernel_segsort.hxx"
#include "memory.hxx"
#include "transform.hxx"

#include "kernel.cuh"
#include "label_propagator.h"
#include "range.cuh"

using namespace mgpu;

template <typename E> void print_device_array(std::string name, E *d_ptr, int len)
{
    vector<int> tmp(len);
    cudaMemcpy(&tmp[0], d_ptr, sizeof(int) * len, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cout << "print device array <<" << name << ">> front: " << len << " elment" << endl;
    for (int i = 0; i < 100; i++)
    {
        cout << tmp[i] << " ";
    }
    cout << endl;
}

// GPU in-core with load imbalance, data-parallel primitives based label propagation
template <typename V, typename E> class InCoreLIDPP : public LabelPropagator<V, E>
{
  public:
    using typename LabelPropagator<V, E>::GraphT;

    InCoreLIDPP(std::shared_ptr<GraphT> _G) : LabelPropagator<V, E>(_G->n), G(_G)
    {
    }
    virtual ~InCoreLIDPP() = default;

    std::pair<double, double> run(int niter);

  private:
    // 保存图数据
    std::shared_ptr<GraphT> G;
    // Methods
    void preprocess();

    int iterate(int i);

    void postprocess();

    int get_count();

    void errorCheck(std::string message);

    void transfer_data();

    void init_gmem(V n, int m);

    void free_gmem();

    int *d_counter;          // 1
    int *d_neighbors;        // m
    int *d_offsets;          // n + 1
    int *d_labels;           // n
    int *d_adj_labels;       // m
    int *d_tmp1;             // m + 1
    int *d_boundaries;       // m + 1
    int *d_boundary_offsets; // n + 1
    int *d_label_weights;    // m
    int *d_tmp_labels;       // n

    mgpu::standard_context_t context; // Used for segmented sort and scan
};

template <typename V, typename E> void InCoreLIDPP<V, E>::errorCheck(std::string message)
{
    auto err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        printf("Error! %s : %s\n", message.c_str(), cudaGetErrorString(err));
    }
}

template <typename V, typename E> void InCoreLIDPP<V, E>::preprocess()
{
    init_gmem(G->n, G->m);
    transfer_data();
}

template <typename V, typename E> std::pair<double, double> InCoreLIDPP<V, E>::run(int niter)
{
    Timer t1, t2;
    t2.start();
    preprocess();

    const int n = G->n;
    const int nthreads = 128;
    const int n_blocks = min(512, divup(n, nthreads));
    initialize_labels<<<n_blocks, nthreads>>>(d_labels, n);
    cudaDeviceSynchronize();
    errorCheck("after initialize label");

    t1.start();
    // Main loop
    for (auto i = 0; i < niter; i++)
    {
        Timer t_iter;
        t_iter.start();
        cout << "iteration: " << niter << " begin" << endl;
        int count = iterate(i);
        cout << "iteration: " << i << " done" << endl;
        t_iter.stop();
        printf("%d: %f (ms), updated %d\n", i + 1, t_iter.elapsed_time(), count);
        //        if (count <= n * 1e-5)
        //        {
        //            break;
        //        }
    }
    // cudaMemcpy(void *, d_labels, sizeof(V) * n, cudaMemcpyDeviceToHost);
    errorCheck("after interation!");
    cudaDeviceSynchronize();
    t1.stop();
    t2.stop();
    postprocess();
    return std::make_pair(t1.elapsed_time(), t2.elapsed_time());
}

template <typename V, typename E> int InCoreLIDPP<V, E>::iterate(int i)
{
    const int n = G->n;
    const int m = G->m;
    cout << "n= " << n << ", m= " << m << endl;

    const int nthreads = 128;
    const int n_blocks = min(512, divup(n, nthreads));
    const int m_blocks = min(512, divup(m, nthreads));
    errorCheck("before lp");
    cout << "n_block = " << n_blocks << " m_blocks =" << m_blocks << endl;

    gather_labels<<<m_blocks, nthreads>>>(d_neighbors, d_labels, d_adj_labels, m);
    cudaDeviceSynchronize();
    // cub segmented sort
    mgpu::standard_context_t ctx;
    Timer t_sort;
    t_sort.start();
    mgpu::segmented_sort(d_adj_labels, m, d_offsets, n, mgpu::less_t<int>(), ctx);
    t_sort.stop();
    cout << "segmented sort elapsed: " << t_sort.elapsed_time() << endl;
  
    Timer t_boundaries;
    t_boundaries.start();
    find_boundaries<<<m_blocks, nthreads>>>(d_adj_labels, m, d_tmp1);
    set_boundary_case<<<n_blocks, nthreads>>>(d_offsets, n, d_tmp1);
    
    t_boundaries.stop();
    cout << "handle boundaries elapsed: " << t_boundaries.elapsed_time() << endl;
  
    Timer t_scan;
    t_scan.start();
    mgpu::scan<mgpu::scan_type_exc>(d_tmp1, m, d_tmp1, ctx);
    t_scan.stop();
    cout << "scan elpased: " << t_scan.elapsed_time() << endl;

    Timer t_scatter;
    t_scatter.start();
    scatter_indexes<<<m_blocks, nthreads>>>(d_tmp1, d_offsets, n, m, d_boundaries, d_boundary_offsets + 1);
    cudaDeviceSynchronize();
    t_scatter.stop();
    cout << "scatter elapsed: " << t_scatter.elapsed_time() << endl;

    // *Load imbalanced*
    Timer t_update;
    t_update.start();
    compute_max_labels<<<n_blocks, nthreads>>>(d_boundary_offsets + 1, d_boundaries, d_adj_labels, d_labels, n,
                                               d_counter);
    cudaDeviceSynchronize();
    errorCheck("incoreli");
    t_update.stop();
    cout << "update label elapsed: " << t_update.elapsed_time() << endl;
    int count = get_count();
    return count;
}

template <typename V, typename E> void InCoreLIDPP<V, E>::postprocess()
{
    free_gmem();
}

template <typename V, typename E> void InCoreLIDPP<V, E>::transfer_data()
{
    cudaMemcpy(d_neighbors, &G->neighbors[0], sizeof(V) * G->m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, &G->offsets[0], sizeof(E) * (G->n + 1), cudaMemcpyHostToDevice);
}

template <typename V, typename E> void InCoreLIDPP<V, E>::init_gmem(V n, int m)
{
    cout << "init gloal mem,args: n=" << n << ",m=" << m << endl;
    cudaMalloc(&d_neighbors, sizeof(int) * m);
    cudaMalloc(&d_offsets, sizeof(int) * (n + 1));
    cudaMalloc(&d_labels, sizeof(int) * n);
    cudaMalloc(&d_adj_labels, sizeof(int) * (m + 1));
    cudaMalloc(&d_tmp1, sizeof(int) * (m + 1));
    cudaMalloc(&d_boundaries, sizeof(int) * (m + 1));
    cudaMalloc(&d_boundary_offsets, sizeof(int) * (n + 1));
    cudaMalloc(&d_label_weights, sizeof(int) * m);
    cudaMalloc(&d_tmp_labels, sizeof(int) * n);
    cudaMalloc(&d_counter, sizeof(int) * 1);

    cudaMemset(d_counter, 0, sizeof(int));
    errorCheck("init_gmem");
}

// Return the number of labels updated
template <typename V, typename E> int InCoreLIDPP<V, E>::get_count()
{
    int counter;
    cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(d_counter, 0, sizeof(int));
    return counter;
}

template <typename V, typename E> void InCoreLIDPP<V, E>::free_gmem()
{
    cudaFree(d_neighbors);
    cudaFree(d_offsets);
    cudaFree(d_labels);
    cudaFree(d_adj_labels);
    cudaFree(d_tmp1);
    cudaFree(d_boundaries);
    cudaFree(d_boundary_offsets);
    cudaFree(d_label_weights);
    cudaFree(d_tmp_labels);
    cudaFree(d_counter);
}
