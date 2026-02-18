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
template<class ValueType, class PointType>
void KDTree<ValueType, PointType>::validate() const {
  auto min = std::numeric_limits<typename PointType::value_type>::lowest();
  auto max = std::numeric_limits<typename PointType::value_type>::max();
  int64_t true_sum_depth = validate(root.get(), 0, {min, min}, {max, max});
  REQUIRE(sum_depth == true_sum_depth);
}

template<class ValueType, class PointType>
int64_t KDTree<ValueType, PointType>::validate(const Node* node, int depth, PointType min, PointType max) const {
  if (!node) {
    return 0;
  }

  REQUIRE(node->depth == depth);
  REQUIRE(node->value.p.x >= min.x);
  REQUIRE(node->value.p.y >= min.y);
  REQUIRE(node->value.p.x < max.x);
  REQUIRE(node->value.p.y < max.y);

  int64_t sum_depth = depth;
  if (depth % 2 == 0) {
    sum_depth += validate(node->children[0].get(), depth + 1, min, {node->value.p.x, max.y});
    sum_depth += validate(node->children[1].get(), depth + 1, {node->value.p.x, min.y}, max);
  } else {
    sum_depth += validate(node->children[0].get(), depth + 1, min, {max.x, node->value.p.y});
    sum_depth += validate(node->children[1].get(), depth + 1, {min.x, node->value.p.y}, max);
  }
  return sum_depth;
}

// Trait structs bundling ValueType, PointType for TEMPLATE_TEST_CASE
template<class V, class P>
struct TreeTraits {
  using ValueType = V;
  using PointType = P;
  using TreeType = KDTree<V, P>;
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
    (TreeTraits<int64_t, Pointi>),
    (TreeTraits<int, Pointi>),
    (TreeTraits<int64_t, Pointf>),
    (TreeTraits<int64_t, Pointd>)
) {
  using Traits    = TestType;
  using TreeType  = typename Traits::TreeType;
  using PointType = typename Traits::PointType;
  using ValueType = typename Traits::ValueType;
  using Value     = typename TreeType::Value;

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
    bool inserted = tree.insert({ValueType(i), p});

    INFO("After:\n" << tree);

    REQUIRE(tree.exists(p));
    if (inserted) {
      REQUIRE(tree.find(p)->value == ValueType(i));
    }

    REQUIRE(inserted == missing);
    tree.validate();
  }

  INFO("Tree: \n" << tree);

  std::vector<Value> values(tree.begin(), tree.end());
  REQUIRE(points.size() == tree.size());
  REQUIRE(points.size() == values.size());

  SECTION("Points are equal") {
    c_sort(points);
    c_sort(values, [](auto a, auto b) { return a.p < b.p; });
    for (size_t i = 0; i < points.size(); i++) {
      REQUIRE(points[i] == values[i].p);
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
      auto value = tree.find(p);
      REQUIRE(bool(value));
      REQUIRE(value->p == p);
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
    for (const Value& v : values) {
      REQUIRE(tree.find_closest(v.p, Norm::L1) == v);
      REQUIRE(tree.find_closest(v.p, Norm::L2) == v);
      REQUIRE(tree.find_closest(v.p, Norm::Linf) == v);
    }
  }

  SECTION("Norm parameter works") {
    // Insert points where L1 and L2 nearest differ
    TreeType norm_tree({
        {1, {10, 0}},
        {2, {9, 4}},
        {3, {7, 7}},
        {4, {11, 11}},
        {5, {-11, -11}},
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
}

template<typename TreeType>
struct BenchFixture {
  using PointType = typename TreeType::point_type;
  using Value     = typename TreeType::Value;

  static constexpr int num_points = 10000;

  std::vector<Value> values;
  std::mt19937 bitgen{Catch::getSeed()};

  BenchFixture() {
    values.reserve(num_points);
    TreeType tree;
    while (static_cast<int>(values.size()) < num_points) {
      Value v(typename TreeType::value_type(values.size()), gen_point());
      if (tree.insert(v)) {
        values.push_back(v);
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
  auto& values   = fix.values;
  auto  gen      = [&fix]{ return fix.gen_point(); };
  constexpr int  N = BenchFixture<TreeType>::num_points;

  BENCHMARK(std::format("[{}] insert {} points", label, N)) {
    TreeType tree;
    for (auto& v : values) tree.insert(v);
  };

  BENCHMARK(std::format("[{}] insert {} points then balance", label, N)) {
    TreeType tree;
    for (auto& v : values) tree.insert(v);
    tree.rebalance();
  };

  BENCHMARK(std::format("[{}] build balanced tree from {} points", label, N)) {
    TreeType tree(values);
  };

  BENCHMARK_ADVANCED(std::format("[{}] iterate into vector", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree(values);
    meter.measure([&tree](int /*i*/) { return std::vector(tree.begin(), tree.end()); });
  };

  BENCHMARK_ADVANCED(std::format("[{}] find", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree(values);
    meter.measure([&tree, &gen](int /*i*/) { return tree.find(gen()); });
  };

  BENCHMARK_ADVANCED(std::format("[{}] find_closest L1", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree(values);
    meter.measure([&tree, &gen](int /*i*/) { return tree.find_closest(gen(), Norm::L1); });
  };

  BENCHMARK_ADVANCED(std::format("[{}] find_closest L2", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree(values);
    meter.measure([&tree, &gen](int /*i*/) { return tree.find_closest(gen(), Norm::L2); });
  };

  BENCHMARK_ADVANCED(std::format("[{}] find_closest Linf", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree(values);
    meter.measure([&tree, &gen](int /*i*/) { return tree.find_closest(gen(), Norm::Linf); });
  };

  BENCHMARK_ADVANCED(std::format("[{}] insert + pop_closest", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree(values);
    meter.measure([&tree, &gen](int i) {
      tree.insert({typename TreeType::value_type(i), gen()});
      return tree.pop_closest(gen());
    });
    REQUIRE(tree.size() > N * 0.9);
  };

  BENCHMARK_ADVANCED(std::format("[{}] rebalance", label))(Catch::Benchmark::Chronometer meter) {
    TreeType tree;
    for (auto& v : values) tree.insert(v);
    meter.measure([&tree](int /*i*/) { tree.rebalance(); return tree.depth_avg(); });
  };
}

TEST_CASE("KDTree Benchmark", "[kdtree][benchmark]") {
  run_benchmarks<KDTreed>("double/int64");
  run_benchmarks<KDTreei>("int/int64");
}
