#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include "kdtree.h"

using namespace kdtree;

void run_size(size_t n, bool first) {
    std::mt19937_64 gen(42);
    
    // 1. KDTreed (Double)
    {
        std::uniform_real_distribution<double> dist(0.0, 1000.0);
        std::vector<Entry<Pointd, int64_t>> data;
        data.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            data.push_back({{dist(gen), dist(gen)}, static_cast<int64_t>(i)});
        }

        auto start_build = std::chrono::high_resolution_clock::now();
        KDTreed tree(data);
        auto end_build = std::chrono::high_resolution_clock::now();
        double build_ms = std::chrono::duration<double, std::milli>(end_build - start_build).count();

        const int num_queries = 1000;
        std::vector<Pointd> queries;
        for(int i=0; i<num_queries; ++i) queries.push_back({dist(gen), dist(gen)});

        auto start_query = std::chrono::high_resolution_clock::now();
        for (const auto& q : queries) {
            auto res = tree.find_closest(q);
            if (res.value < -1000000000) std::cout << "never"; 
        }
        auto end_query = std::chrono::high_resolution_clock::now();
        double query_total_ms = std::chrono::duration<double, std::milli>(end_query - start_query).count();

        if (!first) std::cout << ",\n";
        std::cout << "  {\"test\": \"Core Build\", \"implementation\": \"C++ (double)\", \"n\": " << n << ", \"time_ms\": " << build_ms << ", \"iters\": 1},\n";
        std::cout << "  {\"test\": \"Core Query (1k)\", \"implementation\": \"C++ (double)\", \"n\": " << n << ", \"time_ms\": " << query_total_ms << ", \"iters\": " << num_queries << "}";
        first = false;
    }

    // 2. KDTreei (Int)
    {
        std::uniform_int_distribution<int> dist(0, 1000000); // Larger range for ints
        std::vector<Entry<Pointi, int64_t>> data;
        data.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            data.push_back({{dist(gen), dist(gen)}, static_cast<int64_t>(i)});
        }

        auto start_build = std::chrono::high_resolution_clock::now();
        KDTreei tree(data);
        auto end_build = std::chrono::high_resolution_clock::now();
        double build_ms = std::chrono::duration<double, std::milli>(end_build - start_build).count();

        const int num_queries = 1000;
        std::vector<Pointi> queries;
        for(int i=0; i<num_queries; ++i) queries.push_back({dist(gen), dist(gen)});

        auto start_query = std::chrono::high_resolution_clock::now();
        for (const auto& q : queries) {
            auto res = tree.find_closest(q);
            if (res.value < -1000000000) std::cout << "never"; 
        }
        auto end_query = std::chrono::high_resolution_clock::now();
        double query_total_ms = std::chrono::duration<double, std::milli>(end_query - start_query).count();

        std::cout << ",\n";
        std::cout << "  {\"test\": \"Core Build\", \"implementation\": \"C++ (int)\", \"n\": " << n << ", \"time_ms\": " << build_ms << ", \"iters\": 1},\n";
        std::cout << "  {\"test\": \"Core Query (1k)\", \"implementation\": \"C++ (int)\", \"n\": " << n << ", \"time_ms\": " << query_total_ms << ", \"iters\": " << num_queries << "}";
    }
}

int main() {
    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000, 10000000};
    std::cout << "[\n";
    for (size_t i = 0; i < sizes.size(); ++i) {
        run_size(sizes[i], i == 0);
    }
    std::cout << "\n]\n";
    return 0;
}
