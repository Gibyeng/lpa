//
// Created by Rich on 2021/2/2.
//

#ifndef CUDA_LPA_GRAPH_H
#define CUDA_LPA_GRAPH_H
#include <iostream>
#include <memory>
#include <vector>
using std::cout;
using std::endl;

template <typename T, typename... Args>
std::unique_ptr<T> make_unique_ptr(Args &&...args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
template <typename V, typename E> class Graph {
    // Virtual class representing a graph
    // * V is the type of vertices
    // * E is the type of edges

  public:
    Graph() = default;
    Graph(V _n, E _m) : n(_n), m(_m) {}
    virtual ~Graph() = default;
    V n; // Number of vertices
    E m; // Number of undirected edges
};

// Class representing a graph by the CSR format
template <typename V, typename E> class CSRGraph : public Graph<V, E> {
  public:
    CSRGraph() = default;
    CSRGraph(std::vector<E> _neighbors, std::vector<V> _offsets)
        : Graph<V, E>(_offsets.size() - 1, _neighbors.size()),
          neighbors(_neighbors), offsets(_offsets) {}

    ~CSRGraph() {}
    std::vector<V> neighbors;
    std::vector<E> offsets;
};

template <typename V, typename E> class CGRGraph : public Graph<V, E> {
  public:
    CGRGraph() = default;
    CGRGraph(std::vector<V> &_csr_neighbors, std::vector<E> &_csr_offsets,
             std::vector<V> &_cgr_neighbors, std::vector<E> &_cgr_offsets,
             std::vector<V> &_csr_small_neighbors,
             std::vector<E> &_csr_small_offsets, std::vector<V> &_vertices,
             int sns)
        : Graph<V, E>(_csr_offsets.size() - 1, _csr_neighbors.size()),
          csr_neighbors(_csr_neighbors), csr_offsets(_csr_offsets),
          cgr_neighbors(_cgr_neighbors), cgr_offsets(_cgr_offsets),
          csr_small_neighbors(_csr_small_neighbors),
          csr_small_offsets(_csr_small_offsets), vertices(_vertices),
          small_node_thresh(sns) {
        n_small = _csr_small_offsets.size() - 1;
        m_small = _csr_small_neighbors.size();

        cout << "CGR mixed graph init info\n";
        cout << "csr_neighbors     len: " << _csr_neighbors.size() << endl;
        cout << "csr_offsets       len: " << _csr_offsets.size() << endl;
        cout << "cgr_neighbors     len: " << _cgr_neighbors.size() << endl;
        cout << "cgr_offsets       len: " << _cgr_offsets.size() << endl;
        cout << "csr_small_nbs     len: " << _csr_small_neighbors.size()
             << endl;
        cout << "csr_small_offsets len: " << _csr_small_offsets.size() << endl;
        cout << "_vertices         len: " << _vertices.size() << endl;
        cout << "n_small:               " << n_small << endl;
        cout << "m_small:               " << m_small << endl;
        cout << "n:                     " << _csr_offsets.size() - 1 << endl;
        cout << "m:                     " << _csr_neighbors.size() << endl;
        cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
    }

    ~CGRGraph() {}
    std::vector<V> csr_neighbors;
    std::vector<E> csr_offsets;
    std::vector<V> cgr_neighbors;
    std::vector<E> cgr_offsets;
    std::vector<V> csr_small_neighbors;
    std::vector<E> csr_small_offsets;
    std::vector<V> vertices;
    int small_node_thresh;
    int n_small;
    int m_small;
};

template <typename V, typename E> class CGRGraphPure : public Graph<V, E> {
  public:
    CGRGraphPure() = default;
    CGRGraphPure(std::vector<V> &_csr_neighbors, std::vector<E> &_csr_offsets,
                 std::vector<V> &_cgr_neighbors, std::vector<E> &_cgr_offsets,
                 std::vector<V> &_vertices)
        : Graph<V, E>(_csr_offsets.size() - 1, _csr_neighbors.size()),
          csr_neighbors(_csr_neighbors), csr_offsets(_csr_offsets),
          cgr_neighbors(_cgr_neighbors), cgr_offsets(_cgr_offsets),
          vertices(_vertices) {
        cout << "CGR pure graph info:==========\n";
        cout << "csr_neighbors     len: " << _csr_neighbors.size() << endl;
        cout << "csr_offsets       len: " << _csr_offsets.size() << endl;
        cout << "cgr_neighbors     len: " << _cgr_neighbors.size() << endl;
        cout << "cgr_offsets       len: " << _cgr_offsets.size() << endl;
        cout << "n:                     " << _csr_offsets.size() - 1 << endl;
        cout << "m:                     " << _csr_neighbors.size() << endl;
        cout << "CGR pure graph info:==========\n";
    }

    ~CGRGraphPure() {}
    std::vector<V> csr_neighbors;
    std::vector<E> csr_offsets;
    std::vector<V> cgr_neighbors;
    std::vector<E> cgr_offsets;
    std::vector<V> vertices;
};

#endif // CUDA_LPA_GRAPH_H
