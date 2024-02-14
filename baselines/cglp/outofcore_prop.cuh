#pragma once

#include "../../3rd_party/cub/cub/cub.cuh"
#include "decompressor.cuh"
#include "kernel.cuh"
#include "kernel_scan.hxx"
#include "kernel_segsort.hxx"
#include "label_propagator.h"
#include "memory.hxx"
#include "transform.hxx"
#include "utils.h"
#include <cuda_profiler_api.h>

using std::cout;
using std::endl;
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

template <typename V, typename E> class CGLPPropagator : public LabelPropagator<V, E>
{
  public:
    using typename LabelPropagator<V, E>::CGRGraphTPure;
    CGLPPropagator(std::shared_ptr<CGRGraphTPure> _G) : LabelPropagator<V, E>(_G->n), G(_G)
    {
    }

    std::pair<double, double> run(int niter);

    using LabelPropagator<V, E>::labels;

    // 保存图数据
    std::shared_ptr<CGRGraphTPure> G;

    void preprocess();

    int iterate(int i);

    void postprocess();

    int get_count();

    void errorCheck(std::string message);

    void transfer_data();

    void init_gmem(V n, int m);

    void free_gmem();

  private:
    // GPU memory
    // alloced and transfer data
    // 压缩数据
    int *d_labels; // n

    V *d_cgr_neighbors; // m1
    E *d_cgr_offsets;   // n1
    V *d_vertices;      // n

    //
    E *d_csr_neighbors;
    E *d_csr_offsets;        // n
    int *d_adj_labels;       // m
    int *d_tmp1;             // m + 1
    int *d_boundaries;       // m + 1
    int *d_boundary_offsets; // n + 1
    int *d_label_weights;    // m
    int *d_tmp_labels;       // n

    int *d_counter; // 1

    // For double buffering
    int *d_neighbors_buf;  // B neigbor list
    int *d_offsets_buf;    // n + 1 offset list

    cudaStream_t stream1;
    cudaStream_t stream2;
};

template <typename V, typename E> void CGLPPropagator<V, E>::errorCheck(std::string message)
{
    auto err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        printf("Error! %s : %s\n", message.c_str(), cudaGetErrorString(err));
    }
}

template <typename V, typename E> std::pair<double, double> CGLPPropagator<V, E>::run(int niter)
{
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
    for (auto i = 0; i < niter; i++)
    {
        Timer t_iter;
        t_iter.start();
        cout << "iteration: " << i << " begin" << endl;
        int count = iterate(i);
        cout << "iteration: " << i << " done" << endl;
        t_iter.stop();
        printf("%d: %f (ms), updated %d\n", i + 1, t_iter.elapsed_time(), count);
        // if (count <= n * 1e-5)
        // {
        //     break;
        // }
    }
    cudaMemcpy(labels.get(), d_labels, sizeof(V) * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    t1.stop();
    t2.stop();
    postprocess();
    return std::make_pair(t1.elapsed_time(), t2.elapsed_time());
}

// Return the number of labels updated
template <typename V, typename E> int CGLPPropagator<V, E>::get_count()
{
    int counter;
    cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(d_counter, 0, sizeof(int));
    return counter;
}

template <typename V, typename E> void CGLPPropagator<V, E>::init_gmem(V n, int m)
{
    {
        cout << "GPU MEM ALLOC INFO: " << endl;
        cout << "init gloal mem,args: n=" << n << ",m=" << m << endl;
        cout << "d_labels:         " << G->n << endl;
        cout << "d_csr_offsets:    " << G->csr_offsets.size() << endl;
        cout << "d_csr_neighbors:  " << G->csr_neighbors.size() << endl;
        cout << "d_cgr_neighbors:  " << G->cgr_neighbors.size() << endl;
        cout << "d_cgr_offsets:    " << G->cgr_offsets.size() << endl;
        cout << "d_vertices:       " << G->vertices.size() << endl;
        cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
    }
    cudaMalloc(&d_labels, sizeof(int) * G->n);

    cudaMalloc(&d_csr_offsets, sizeof(int) * G->csr_offsets.size());
    cudaMalloc(&d_csr_neighbors, sizeof(int) * G->csr_neighbors.size());
    cudaMalloc(&d_adj_labels, sizeof(int) * (m + 1));
    cudaMalloc(&d_tmp1, sizeof(int) * (m + 1));
    cudaMalloc(&d_boundaries, sizeof(int) * (m + 1));
    cudaMalloc(&d_boundary_offsets, sizeof(int) * (n + 1));
    cudaMalloc(&d_label_weights, sizeof(int) * m);
    cudaMalloc(&d_tmp_labels, sizeof(int) * n);

    cudaMalloc(&d_cgr_neighbors, sizeof(int) * G->cgr_neighbors.size());

    cudaMalloc(&d_cgr_offsets, sizeof(int) * G->cgr_offsets.size());

    cudaMalloc(&d_vertices, sizeof(int) * G->vertices.size());
    cudaMalloc(&d_counter, sizeof(int) * 1);

    cudaMemset(d_counter, 0, sizeof(int));

    errorCheck("GPU mem alloc");
}

template <typename V, typename E> void CGLPPropagator<V, E>::free_gmem()
{
    cudaFree(d_labels);
    cudaFree(d_csr_offsets);
    cudaFree(d_csr_neighbors);
    cudaFree(d_adj_labels);
    cudaFree(d_tmp1);
    cudaFree(d_boundaries);
    cudaFree(d_boundary_offsets);
    cudaFree(d_label_weights);
    cudaFree(d_tmp_labels);
    cudaFree(d_cgr_neighbors);
    cudaFree(d_cgr_offsets);
    cudaFree(d_vertices);
    cudaFree(d_counter);
    errorCheck("GPU mem free");
    return;
}

// template <typename V, typename E> void CGLPPropagator<V, E>::preprocess()
// {
//     init_gmem(G->n, G->m);
//     transfer_data();
// }

template<typename V, typename E, typename S>
void AsyncLP<V, E, S>::preprocess()
{
    init_gmem(G->n, this->B);

    this->compute_batch_boundaries();

    stream1 = this->context->Stream();
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
}


template<typename V, typename E, typename S>
int AsyncLP<V, E, S>::iterate(int i)
{
    if (i == 0) {
        // The first batch
        this->transfer_batch(0, this->d_neighbors, this->d_offsets, stream1);
        cudaDeviceSynchronize();
    }

    int nbatches = this->get_num_batches();
    for (auto j: range(nbatches)) {
        int batch_n = this->get_num_batch_vertices(j);
        int batch_m = this->get_num_batch_edges(j);

        this->transfer_next_batch(j, d_neighbors_buf, d_offsets_buf, stream2);

        // this->perform_lp(batch_n, batch_m, this->bbs[j], &this->h_norm_offsets[G->n + 1], stream1);
        cudaDeviceSynchronize();
        swap_buffers();
    }
    int count = this->get_count();
    return count;
}

// template <typename V, typename E> int CGLPPropagator<V, E>::perform_lp(int i)
// {
//     int num_big = 0;
//     int num_medium = 0;
//     // record the index in vectices array.
//     int big_index_end = 0;
//     int medium_index_end = 0;
//     const SIZE_TYPE medium = 32;
//     const SIZE_TYPE big = 128;

//     // computer nodes size
//     big_index_end = binary_search(SIZE_TYPE(0), G->n - 1, G->csr_offsets, G->vertices, big);
//     printf("\nbig_index %d \n", big_index_end);
//     medium_index_end =
//         medium == 0 ? G->n - 1 : binary_search(SIZE_TYPE(0), G->n - 1, G->csr_offsets, G->vertices, medium);

//     printf("medium_index %d \n", medium_index_end);
//     num_big = big_index_end;
//     num_medium = medium_index_end - big_index_end;

//     printf("num_big: %d,num_medium: %d,\n", num_big, num_medium);

//     int block_num = (num_big < 512) ? num_big : 512;
//     printf("block num for big nodes :%d\n", block_num);
//     block_num = (num_medium < 512) ? divup(num_medium, 4) : 512;
//     printf("block num for mid nodes :%d\n", block_num);

//     errorCheck("before LP!");
//     Timer t_all;
//     t_all.start();
//     // launch big node kernel
//     if (num_big > 0)
//     {
//         cout << "BIG NODE LPA starts: " << endl;
//         Timer t_big;
//         t_big.start();
//         decompress_large_vertex<<<block_num, 128>>>(big_index_end, d_cgr_neighbors, d_vertices, d_cgr_offsets,
//                                                     d_csr_offsets);
//         cudaDeviceSynchronize();
//         t_big.stop();

//         errorCheck("after bit nodes LP!");
//         cout << "compute big kernel time:" << t_big.elapsed_time() << endl;
//         cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
//     }

//     // launch medium node kernel
//     if (num_medium > 0)
//     {
//         cout << "MID NODE LPA starts: " << endl;
//         Timer t_mid;
//         t_mid.start();

//         decompress_mid_vertex<<<block_num, 128>>>(big_index_end, medium_index_end, d_cgr_neighbors, d_vertices,
//                                                   d_cgr_offsets, d_csr_offsets);
//         cudaDeviceSynchronize();
//         errorCheck("after mid nodes LP!");
//         t_mid.stop();
//         cout << "compute mid kernel time :" << t_mid.elapsed_time() << endl;
//         cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
//     }
//     // start to propagate
//     Timer t_lp;
//     t_lp.start();
//     cout << "cglp lp starts: " << endl;
//     const int n = G->n;
//     const int m = G->m;
//     cout << "n= " << n << ", m= " << m << endl;

//     const int nthreads = 128;
//     const int n_blocks = min(512, divup(n, nthreads));
//     const int m_blocks = min(512, divup(m, nthreads));
//     cout << "n_block = " << n_blocks << " m_blocks =" << m_blocks << endl;

//     gather_labels<<<m_blocks, nthreads>>>(d_csr_neighbors, d_labels, d_adj_labels, m);

//     cudaDeviceSynchronize();
//     // print_device_array("gather_labels", d_adj_labels, 100);

//     // t_gather.stop();
//     // cub segmented sort
//     mgpu::standard_context_t ctx;
//     Timer t_sort;
//     t_sort.start();
//     mgpu::segmented_sort(d_adj_labels, m, d_csr_offsets, n, mgpu::less_t<int>(), ctx);

//     t_sort.stop();
//     cout << "segmented sort elapsed: " << t_sort.elapsed_time() << endl;
//     // print_device_array("sorted_gather_labels", d_adj_labels, 100);
//     Timer t_boundaries;
//     t_boundaries.start();
//     find_boundaries<<<m_blocks, nthreads>>>(d_adj_labels, m, d_tmp1);

//     set_boundary_case<<<n_blocks, nthreads>>>(d_csr_offsets, n, d_tmp1);
//     cudaDeviceSynchronize();
//     t_boundaries.stop();
//     cout << "handle boundaries elapsed: " << t_boundaries.elapsed_time() << endl;
//     // print_device_array("boundaries", d_tmp1, 200);
//     // mgpu::scan(this->d_tmp1, G->m + 1, *this->context);
//     Timer t_scan;
//     t_scan.start();
//     mgpu::scan<mgpu::scan_type_exc>(d_tmp1, m, d_tmp1, ctx);
//     t_scan.stop();
//     cout << "scan elpased: " << t_scan.elapsed_time() << endl;
//     // print_device_array("scan_tmp1", d_tmp1, 100);
//     Timer t_scatter;
//     t_scatter.start();
//     scatter_indexes<<<m_blocks, nthreads>>>(d_tmp1, d_csr_offsets, n, m, d_boundaries, d_boundary_offsets + 1);
//     cudaDeviceSynchronize();
//     t_scatter.stop();
//     cout << "scatter elapsed: " << t_scatter.elapsed_time() << endl;
//     // print_device_array("d_boundaries", d_boundaries, 100);
//     // print_device_array("d_boundaries_offsets", d_boundary_offsets, 100);
//     errorCheck("before compute max label");
//     // *Load imbalanced*
//     Timer t_update;
//     t_update.start();
//     // sge -> d_b_o sge_off -> d_b
//     compute_max_labels<<<n_blocks, nthreads>>>(d_boundary_offsets + 1, d_boundaries, d_adj_labels, d_labels, n,
//                                                d_counter);
//     cudaDeviceSynchronize();
//     errorCheck("cglp lp");
//     t_update.stop();
//     cout << "update label elapsed: " << t_update.elapsed_time() << endl;

//     t_all.stop();
//     cout << "cglp elapsed: " << t_all.elapsed_time() << endl;
//     cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
//     return big_index_end;
// }

// template <typename V, typename E> void CGLPPropagator<V, E>::postprocess()
// {
//     free_gmem();
// }

template<typename V, typename E, typename S>
void AsyncLP<V, E, S>::postprocess()
{
    free_gmem();

    cudaStreamDestroy(stream2);
}

template <typename V, typename E> void CGLPPropagator<V, E>::transfer_data()
{
    cout << "Date transfer,host2device: " << endl;
    Timer t;
    int n = G->vertices.size();
    int m1 = G->cgr_neighbors.size();
    int n1 = G->cgr_offsets.size();

    t.start();
    // 需要传输的数据为： n + m1 + n1 + m2+ n2+ n= 3n + m1 + m2
    cudaMemcpy(d_csr_offsets, &G->csr_offsets[0], sizeof(V) * G->csr_offsets.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_neighbors, &G->csr_neighbors[0], sizeof(E) * G->csr_neighbors.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cgr_neighbors, &G->cgr_neighbors[0], sizeof(V) * m1, cudaMemcpyHostToDevice);

    cudaMemcpy(d_cgr_offsets, &G->cgr_offsets[0], sizeof(V) * n1, cudaMemcpyHostToDevice);

    cudaMemcpy(d_vertices, &G->vertices[0], sizeof(V) * n, cudaMemcpyHostToDevice);

    t.stop();
    cout << "transfer data elapse: " << t.elapsed_time() << endl;
    cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
}
