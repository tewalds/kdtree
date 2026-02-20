#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include "kdtree.h"

using namespace kdtree;

struct PointRaw {
    double lat, lon;
};

std::vector<PointRaw> read_tsv(const std::string& path) {
    std::vector<PointRaw> points;
    std::ifstream file(path);
    double lat, lon;
    while (file >> lat >> lon) {
        points.push_back({lat, lon});
    }
    return points;
}

int main() {
    std::string path = "benchmarks/cdt.tsv";
    auto points = read_tsv(path);
    if (points.empty()) return 1;

    int n = points.size();
    std::cout << "[\n";

    // 1. Bulk
    std::vector<Entry<Pointd, int64_t>> entries;
    entries.reserve(n);
    for (int i = 0; i < n; ++i) entries.push_back({{points[i].lat, points[i].lon}, i});
    
    auto start = std::chrono::high_resolution_clock::now();
    KDTreed tree_bulk(entries);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "  {\"test\": \"GPX Bulk Build\", \"implementation\": \"C++\", \"n\": " << n << ", \"time_ms\": " << std::chrono::duration<double, std::milli>(end-start).count() << "},\n";

    // 2. Inc Load
    start = std::chrono::high_resolution_clock::now();
    KDTreed tree_inc;
    for (int i = 0; i < n; ++i) tree_inc.insert({points[i].lat, points[i].lon}, i);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "  {\"test\": \"GPX Inc Load\", \"implementation\": \"C++\", \"n\": " << n << ", \"time_ms\": " << std::chrono::duration<double, std::milli>(end-start).count() << "},\n";

    // 3. Dedupe
    struct { std::string label; double val; } thresholds[] = {{"1m", 0.00001}, {"10m", 0.0001}, {"100m", 0.001}};
    for (int i=0; i<3; ++i) {
        start = std::chrono::high_resolution_clock::now();
        KDTreed tree_dedupe;
        int kept = 0;
        for (const auto& p : points) {
            if (!tree_dedupe.find_closest(Pointd{p.lat, p.lon}, L2sq{}, thresholds[i].val * thresholds[i].val)) {
                tree_dedupe.insert({p.lat, p.lon}, kept++);
            }
        }
        end = std::chrono::high_resolution_clock::now();
        std::cout << "  {\"test\": \"GPX Dedupe (" << thresholds[i].label << ")\", \"implementation\": \"C++\", \"n\": " << n << ", \"time_ms\": " << std::chrono::duration<double, std::milli>(end-start).count() << "}" << (i == 2 ? "" : ",") << "\n";
    }

    std::cout << "]\n";
    return 0;
}
