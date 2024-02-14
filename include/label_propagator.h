// -*- coding: utf-8 -*-

#pragma once

#include <memory>
#include <random>
#include <cstdlib>
#include "graph.h"
#include "utils.h"

template <typename V, typename E>
class LabelPropagator
{
public:
    using GraphT = CSRGraph<V, E>;
    using CGRGraphT = CGRGraph<V, E>;
    using CGRGraphTPure = CGRGraphPure<V, E>;
    using CommonGraph = Graph<V, E>;
    LabelPropagator() = default;
    LabelPropagator(int n) : labels(new V[n]) {}
    LabelPropagator(std::shared_ptr<GraphT> _G)
        : G(_G), labels(new V[_G->n]) {}

    virtual ~LabelPropagator() = default;

    virtual std::pair<double, double> run(int niter);

    std::vector<V> get_labels()
    {
        auto p = labels.get();
        std::vector<V> tmp_labels(p, p + G->n);
        return tmp_labels;
    }

    // protected:
    std::shared_ptr<GraphT> G;
    std::unique_ptr<V[]> labels;
};

template <typename V, typename E>
std::pair<double, double> LabelPropagator<V, E>::run(int niter)
{
    Timer t;
    t.start();

    std::vector<V> vertices(G->n);

    t.stop();

    auto f = t.elapsed_time();
    return {f, f};
}