#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include "kdtree.h"

using namespace kdtree;

template <typename Metric>
void run_metric_bench(const KDTreed& tree, const std::vector<Pointd>& queries,
                      const std::string& metric_name, const Metric& metric, int n) {
    const int num_queries = queries.size();

    // 1. find_closest
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& q : queries) {
        tree.find_closest(q, metric, -1.0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  {\"test\": \"Query: find_closest\", \"implementation\": \"C++ (" << metric_name << ")\", \"n\": " << n << ", \"time_ms\": " << ms << ", \"iters\": " << num_queries << "},\n";

    // 2. find_closest_k (k=5)
    start = std::chrono::high_resolution_clock::now();
    for (const auto& q : queries) {
        tree.find_closest_k(q, 5, metric, -1.0);
    }
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  {\"test\": \"Query: find_closest_k=5\", \"implementation\": \"C++ (" << metric_name << ")\", \"n\": " << n << ", \"time_ms\": " << ms << ", \"iters\": " << num_queries << "},\n";

    // 3. find_all_within
    double radius = 10.0;
    if (metric_name == "GreatCircle") {
        radius = 100000.0; // 100km
    } else if (metric_name.find("L2sq") != std::string::npos) {
        radius = radius * radius;
    }

    start = std::chrono::high_resolution_clock::now();
    for (const auto& q : queries) {
        tree.find_all_within(q, metric, radius);
    }
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  {\"test\": \"Query: find_all_within\", \"implementation\": \"C++ (" << metric_name << ")\", \"n\": " << n << ", \"time_ms\": " << ms << ", \"iters\": " << num_queries << "}";
}

int main() {
    std::vector<int> sizes = {1000, 10000, 100000, 1000000};
    std::cout << "[\n";
    bool first_outer = true;

    for (int n : sizes) {
        std::mt19937_64 gen(42);
        std::uniform_real_distribution<double> dist(0.0, 1000.0);
        std::vector<Entry<Pointd, int64_t>> data;
        data.reserve(n);
        for (int i = 0; i < n; ++i) data.push_back({{dist(gen), dist(gen)}, i});

        KDTreed tree(data);

        std::vector<Pointd> queries;
        for (int i = 0; i < 1000; ++i) queries.push_back({dist(gen), dist(gen)});

        if (!first_outer) std::cout << ",\n";

        run_metric_bench(tree, queries, "L1", L1{}, n); std::cout << ",\n";
        run_metric_bench(tree, queries, "L2", L2{}, n); std::cout << ",\n";
        run_metric_bench(tree, queries, "L2sq", L2sq{}, n); std::cout << ",\n";
        run_metric_bench(tree, queries, "Linf", Linf{}, n); std::cout << ",\n";
        run_metric_bench(tree, queries, "ToroidalL1", Toroidal<L1, double>{{1000, 1000}}, n); std::cout << ",\n";
        run_metric_bench(tree, queries, "ToroidalL2", Toroidal<L2, double>{{1000, 1000}}, n); std::cout << ",\n";
        run_metric_bench(tree, queries, "ToroidalL2sq", Toroidal<L2sq, double>{{1000, 1000}}, n); std::cout << ",\n";
        run_metric_bench(tree, queries, "ToroidalLinf", Toroidal<Linf, double>{{1000, 1000}}, n); std::cout << ",\n";
        run_metric_bench(tree, queries, "GreatCircle", GreatCircle{6371000.0}, n);

        first_outer = false;
    }

    std::cout << "\n]\n";
    return 0;
}
