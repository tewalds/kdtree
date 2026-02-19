#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include "kdtree.h"

using namespace kdtree;

struct BenchResult {
    size_t n;
    double build_time_ms;
    double query_time_us; // Mean time for one NN query
};

BenchResult run_bench(size_t n) {
    std::mt19937_64 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1000.0);

    std::vector<Entry<Pointd, int64_t>> data;
    data.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        data.push_back({{dist(gen), dist(gen)}, static_cast<int64_t>(i)});
    }

    // Measure Build
    auto start_build = std::chrono::high_resolution_clock::now();
    KDTreed tree(data);
    auto end_build = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(end_build - start_build).count();

    // Measure Queries (batch of 1000 to get a good average)
    const int num_queries = 1000;
    std::vector<Pointd> queries;
    for(int i=0; i<num_queries; ++i) queries.push_back({dist(gen), dist(gen)});

    auto start_query = std::chrono::high_resolution_clock::now();
    for (const auto& q : queries) {
        auto res = tree.find_closest(q);
        // Volatile-ish check to prevent optimization
        if (res.value < -1000000000) std::cout << "never"; 
    }
    auto end_query = std::chrono::high_resolution_clock::now();
    double query_us = std::chrono::duration<double, std::micro>(end_query - start_query).count() / num_queries;

    return {n, build_ms, query_us};
}

int main(int argc, char** argv) {
    std::vector<size_t> sizes = {10000, 100000, 1000000};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) sizes.push_back(std::stoull(argv[i]));
    }

    std::cout << "{\n  \"results\": [\n";
    for (size_t i = 0; i < sizes.size(); ++i) {
        auto res = run_bench(sizes[i]);
        std::cout << "    {\n"
                  << "      \"n\": " << res.n << ",\n"
                  << "      \"build_ms\": " << res.build_time_ms << ",\n"
                  << "      \"query_us\": " << res.query_time_us << "\n"
                  << "    }" << (i == sizes.size() - 1 ? "" : ",") << "\n";
    }
    std::cout << "  ]\n}\n";

    return 0;
}
