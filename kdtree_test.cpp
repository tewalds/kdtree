// Copyright 2024-2025
// Licensed under the Apache License, Version 2.0

#include <algorithm>
#include <format>
#include <iostream>
#include <random>

#include "catch2/catch_amalgamated.hpp"
#include "kdtree.h"

using namespace kdtree;

template <typename Container, typename T>
bool c_linear_search(const Container& c, const T& value) {
    return std::find(std::begin(c), std::end(c), value) != std::end(c);
}

template <typename Container>
void c_sort(Container& c) {
    std::sort(std::begin(c), std::end(c));
}

template <typename Container, typename Compare>
void c_sort(Container& c, Compare comp) {
    std::sort(std::begin(c), std::end(c), comp);
}

template <typename Container, typename T>
auto c_find(Container& c, const T& value) {
    return std::find(std::begin(c), std::end(c), value);
}

// Implement validate() for all template instantiations
template<class PointType, class ValueType>
void KDTree<PointType, ValueType>::validate() const {
  auto min = std::numeric_limits<typename PointType::value_type>::lowest();
  auto max = std::numeric_limits<typename PointType::value_type>::max();
  int64_t true_sum_depth = validate(root.get(), 0, {min, min}, {max, max});
  REQUIRE(sum_depth == true_sum_depth);
}

template<class PointType, class ValueType>
int64_t KDTree<PointType, ValueType>::validate(const Node* node, int depth, PointType min, PointType max) const {
  if (!node) {
    return 0;
  }

  REQUIRE(node->depth == depth);
  REQUIRE(node->entry.p.x >= min.x);
  REQUIRE(node->entry.p.y >= min.y);
  REQUIRE(node->entry.p.x < max.x);
  REQUIRE(node->entry.p.y < max.y);

  int64_t sum_depth = depth;
  if (depth % 2 == 0) {
    sum_depth += validate(node->children[0].get(), depth + 1, min, {node->entry.p.x, max.y});
    sum_depth += validate(node->children[1].get(), depth + 1, {node->entry.p.x, min.y}, max);
  } else {
    sum_depth += validate(node->children[0].get(), depth + 1, min, {max.x, node->entry.p.y});
    sum_depth += validate(node->children[1].get(), depth + 1, {min.x, node->entry.p.y}, max);
  }
  return sum_depth;
}

// Trait structs bundling PointType, ValueType for TEMPLATE_TEST_CASE
template<class P, class V>
struct TreeTraits {
  using PointType = P;
  using ValueType = V;
  using TreeType = KDTree<P, V>;
};

// Helper to generate a random point with the right distribution for the coord type
template<class PointType>
PointType gen_point_in_range(std::mt19937& bitgen, int lo, int hi) {
  using T = typename PointType::value_type;
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> d(lo, hi);
    return PointType(d(bitgen), d(bitgen));
  } else {
    std::uniform_real_distribution<T> d(lo, hi);
    return PointType(d(bitgen), d(bitgen));
  }
}

TEMPLATE_TEST_CASE("KDTree", "[kdtree]",
    (TreeTraits<Pointi, int64_t>),
    (TreeTraits<Pointi, int>),
    (TreeTraits<Pointf, int64_t>),
    (TreeTraits<Pointd, int64_t>)
) {
  using Traits    = TestType;
  using TreeType  = typename Traits::TreeType;
  using PointType = typename Traits::PointType;
  using ValueType = typename Traits::ValueType;
  using Entry     = typename TreeType::Entry;

  TreeType tree;
  std::vector<PointType> points;
  std::mt19937 bitgen(Catch::getSeed());

  for (int i = 0; i < 50; i++) {
    PointType p = gen_point_in_range<PointType>(bitgen, 0, 9);
    bool missing = !c_linear_search(points, p);
    if (missing) {
      points.push_back(p);
    }
    CAPTURE(points, i, p);
    INFO("Before:\n" << tree);
    bool inserted = tree.insert({p, ValueType(i)});

    INFO("After:\n" << tree);

    REQUIRE(tree.exists(p));
    if (inserted) {
      REQUIRE(tree.find(p)->value == ValueType(i));
    }

    REQUIRE(inserted == missing);
    tree.validate();
  }

  INFO("Tree: \n" << tree);

  std::vector<Entry> entries(tree.begin(), tree.end());
  REQUIRE(points.size() == tree.size());
  REQUIRE(points.size() == entries.size());

  SECTION("Points are equal") {
    c_sort(points);
    c_sort(entries, [](auto a, auto b) { return a.p < b.p; });
    for (size_t i = 0; i < points.size(); i++) {
      REQUIRE(points[i] == entries[i].p);
    }
  }

  SECTION("Rebalance") {
    INFO("Before: " << tree.balance_str() << tree);
    tree.rebalance();
    INFO("After:  " << tree.balance_str() << tree);
    tree.validate();
  }

  SECTION("Find/exists works") {
    // Probe every point we inserted (positive cases)
    for (const PointType& p : points) {
      INFO("Find " << p << "\n" << tree);
      REQUIRE(tree.exists(p));
      auto entry = tree.find(p);
      REQUIRE(bool(entry));
      REQUIRE(entry->p == p);
    }
    // Also probe integer-coord points in [0,9]^2 — covers non-existent values for
    // all types, and for Pointi also exercises the positive cases via a fixed grid
    // order that's independent of insertion order.
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        PointType p(x, y);
        bool exists = c_linear_search(points, p);
        INFO("Find " << p << ", exists: " << exists << "\n" << tree);
        REQUIRE(exists == tree.exists(p));
        REQUIRE(exists == bool(tree.find(p)));
      }
    }
  }

  SECTION("Finding a known value returns that value") {
    for (const Entry& e : entries) {
      REQUIRE(tree.find_closest(e.p, Norm::L1) == e);
      REQUIRE(tree.find_closest(e.p, Norm::L2) == e);
      REQUIRE(tree.find_closest(e.p, Norm::Linf) == e);
    }
  }

  SECTION("Norm parameter works") {
    // Insert points where L1 and L2 nearest differ
    TreeType norm_tree({
        {{10, 0}, 1},
        {{9, 4}, 2},
        {{7, 7}, 3},
        {{11, 11}, 4},
        {{-11, -11}, 5},
    });

    REQUIRE(norm_tree.find_closest({0, 0}, Norm::L1).value == 1);
    REQUIRE(norm_tree.find_closest({0, 0}, Norm::L2).value == 2);
    REQUIRE(norm_tree.find_closest({0, 0}, Norm::Linf).value == 3);
  }

  SECTION("Remove works") {
    // Iterate a fixed grid that's independent of insertion order, covering both
    // points that exist in the tree and many that don't — verifying that removing
    // a non-existent point returns false and leaves the tree unchanged.
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        PointType p(x, y);
        INFO("Remove " << p << "\n" << tree);
        bool exists = c_linear_search(points, p);
        size_t size = tree.size();
        REQUIRE(exists == tree.remove(p));
        REQUIRE(tree.size() == size - (exists ? 1 : 0));
        REQUIRE(!tree.exists(p));
        INFO("after\n" << tree);
        tree.validate();
      }
    }
    // All integer-coord points in [0,9]^2 have been removed; for Pointi the tree
    // must now be empty. For float/double, inserted points won't be integer-valued
    // so the grid removes nothing — remove the remaining points explicitly.
    for (const PointType& p : points) {
      tree.remove(p);
    }
    REQUIRE(tree.empty());
  }

  SECTION("pop works") {
    REQUIRE(tree.size() == points.size());
    while (!tree.empty()) {
      tree.pop_closest(PointType(3, 4));
      tree.validate();
    }
  }

  SECTION("pop with norm works") {
    REQUIRE(tree.size() == points.size());
    int count = 0;
    while (!tree.empty() && count < 10) {
      tree.pop_closest(PointType(3, 4), Norm::L1);
      tree.validate();
      count++;
    }
  }

  SECTION("find_closest with max_dist works") {
    TreeType tree({
        {{0, 0}, 1},
        {{10, 10}, 2}
    });

    // L1 (Manhattan): dist({1,1}, {0,0}) = 1+1 = 2
    REQUIRE(tree.find_closest({1, 1}, 3, Norm::L1).has_value());
    REQUIRE(tree.find_closest({1, 1}, 2, Norm::L1).has_value());
    REQUIRE(!tree.find_closest({1, 1}, 1, Norm::L1).has_value());

    // L2 (Euclidian): dist({3,4}, {0,0}) = 5
    REQUIRE(tree.find_closest({3, 4}, 6, Norm::L2).has_value());
    REQUIRE(tree.find_closest({3, 4}, 5, Norm::L2).has_value());
    REQUIRE(!tree.find_closest({3, 4}, 4, Norm::L2).has_value());

    // Linf (Chebyshev): dist({2,2}, {0,0}) = max(2,2) = 2
    REQUIRE(tree.find_closest({2, 2}, 3, Norm::Linf).has_value());
    REQUIRE(tree.find_closest({2, 2}, 2, Norm::Linf).has_value());
    REQUIRE(!tree.find_closest({2, 2}, 1, Norm::Linf).has_value());
  }

  SECTION("Structured bindings") {
    for (auto [p, v] : tree) {
      REQUIRE(tree.find(p)->value == v);
    }
  }
}

template<typename TreeType>
struct BenchFixture {
  using PointType = typename TreeType::point_type;
  using Entry     = typename TreeType::Entry;

  static constexpr int num_points = 10000;

  std::vector<Entry> entries;
  std::mt19937 bitgen{Catch::getSeed()};

  BenchFixture() {
    entries.reserve(num_points);
    TreeType tree;
    while (static_cast<int>(entries.size()) < num_points) {
      Entry e(gen_point(), typename TreeType::value_type(entries.size()));
      if (tree.insert(e)) {
        entries.push_back(e);
      }
    }
  }

  PointType gen_point() {
    return gen_point_in_range<PointType>(bitgen, 0, 4000);
  }
};

template<typename TreeType>
void run_benchmarks(const std::string& label) {
  BenchFixture<TreeType> fix;
  auto& entries  = fix.entries;
  auto  gen      = [&fix]{ return fix.gen_point(); };
  constexpr int  N = BenchFixture<TreeType>::num_points;

  BENCHMARK(std::format("[{}] insert {} points", label, N)) {
    TreeType tree;
    for (auto& e : entries) tree.insert(e);
  };

  BENCHMARK(std::format("[{}] insert {} points then balance", label, N)) {
    TreeType tree;
    for (auto& e : entries) tree.insert(e);
    tree.rebalance();
  };

  BENCHMARK(std::format("[{}] build balanced tree from {} points", label, N)) {
    TreeType tree(entries);
  };

  BENCHMARK_ADVANCED(std::format("[{}] iterate into vector", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree(entries);
    meter.measure([&tree](int /*i*/) { return std::vector(tree.begin(), tree.end()); });
  };

  BENCHMARK_ADVANCED(std::format("[{}] find", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree(entries);
    meter.measure([&tree, &gen](int /*i*/) { return tree.find(gen()); });
  };

  BENCHMARK_ADVANCED(std::format("[{}] find_closest L1", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree(entries);
    meter.measure([&tree, &gen](int /*i*/) { return tree.find_closest(gen(), Norm::L1); });
  };

  BENCHMARK_ADVANCED(std::format("[{}] find_closest L2", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree(entries);
    meter.measure([&tree, &gen](int /*i*/) { return tree.find_closest(gen(), Norm::L2); });
  };

  BENCHMARK_ADVANCED(std::format("[{}] find_closest Linf", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree(entries);
    meter.measure([&tree, &gen](int /*i*/) { return tree.find_closest(gen(), Norm::Linf); });
  };

  BENCHMARK_ADVANCED(std::format("[{}] insert + pop_closest", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree(entries);
    meter.measure([&tree, &gen](int i) {
      tree.insert(gen(), typename TreeType::value_type(i));
      return tree.pop_closest(gen());
    });
    REQUIRE(tree.size() > N * 0.9);
  };

  BENCHMARK_ADVANCED(std::format("[{}] rebalance", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree;
    for (auto& e : entries) tree.insert(e);
    meter.measure([&tree](int /*i*/) { tree.rebalance(); return tree.depth_avg(); });
  };
}

TEST_CASE("KDTree Benchmark", "[kdtree][benchmark]") {
  run_benchmarks<KDTreed>("double/int64");
  run_benchmarks<KDTreei>("int/int64");
}
