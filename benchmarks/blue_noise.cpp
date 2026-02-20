// https://blog.demofox.org/2017/10/20/generating-blue-noise-sample-points-with-mitchells-best-candidate-algorithm/

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include "kdtree.h"

using namespace kdtree;

void run_bench(int n, bool first) {
    Pointd bounds(1000.0, 1000.0);
    Toroidal<L2sq, double> metric{bounds};
    KDTreed tree;

    std::mt19937_64 gen(42);
    std::uniform_real_distribution<double> dist_x(0.0, bounds.x);
    std::uniform_real_distribution<double> dist_y(0.0, bounds.y);
    auto rand_p = [&]() { return Pointd(dist_x(gen), dist_y(gen)); };

    auto start = std::chrono::high_resolution_clock::now();
    tree.insert(rand_p(), 0);

    double total_query_ms = 0;
    int query_count = 0;

    for (int i = 1; i < n; ++i) {
        int num_candidates = i + 1;
        Pointd best_p;
        double max_sq_dist = -1.0;

        for (int j = 0; j < num_candidates; ++j) {
            Pointd p = rand_p();
            auto q_start = std::chrono::high_resolution_clock::now();
            auto closest = tree.find_closest(p, metric, -1.0);
            auto q_end = std::chrono::high_resolution_clock::now();

            total_query_ms += std::chrono::duration<double, std::milli>(q_end - q_start).count();
            query_count++;

            double d2 = metric.dist(p, closest->p);
            if (d2 > max_sq_dist) {
                max_sq_dist = d2;
                best_p = p;
            }
        }
        tree.insert(best_p, i);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (!first) std::cout << ",\n";
    std::cout << "  {\"test\": \"Blue Noise Total\", \"implementation\": \"C++\", \"n\": " << n << ", \"time_ms\": " << total_ms << ", \"iters\": 1},\n";
    std::cout << "  {\"test\": \"Blue Noise Search\", \"implementation\": \"C++\", \"n\": " << n << ", \"time_ms\": " << total_query_ms << ", \"iters\": " << query_count << "}";
}

int main() {
    std::vector<int> sizes = {500, 1000, 2000};
    std::cout << "[\n";
    for (size_t i = 0; i < sizes.size(); ++i) {
        run_bench(sizes[i], i == 0);
    }
    std::cout << "\n]\n";
    return 0;
}
