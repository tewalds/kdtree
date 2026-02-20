// Copyright 2024-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <concepts>

namespace kdtree {

template <typename T>
struct Point {
  using value_type = T;

  union {
    struct { T x, y; };
    T coords[2];
  };

  Point() : x(0), y(0) {}
  Point(T x_, T y_) : x(x_), y(y_) {}

  bool operator==(const Point& o) const { return x == o.x && y == o.y; }
  bool operator!=(const Point& o) const { return !(*this == o); }
  bool operator<(const Point& o) const { return x != o.x ? x < o.x : y < o.y; }

  friend std::ostream& operator<<(std::ostream& stream, const Point& p) {
    return stream << "{" << p.x << ", " << p.y << "}";
  }
};

template <typename P>
concept IsPoint = requires(P p) {
  { p.coords[0] };
  { p.x };
  { p.y };
};

template <typename M, typename P1, typename P2>
concept IsMetric = requires(M m, P1 p1, P2 p2) {
  { m.dist(p1, p2) };
  { m.axis_dist(p1, p2, 0) };
};

using Pointi = Point<int>;
using Pointf = Point<float>;
using Pointd = Point<double>;

template <IsPoint PointType, typename ValueType>
struct Entry {
  PointType p;
  ValueType value;

  Entry() = default;
  Entry(PointType pt, ValueType v) : p(pt), value(v) {}

  bool operator==(const Entry& o) const = default;
  bool operator!=(const Entry& o) const = default;

  friend std::ostream& operator<<(std::ostream& stream, const Entry& e) {
    return stream << "Entry(" << e.p << ", " << e.value << ")";
  }
};

struct L1 {
  auto dist(const IsPoint auto& a, const IsPoint auto& b) const {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
  }
  auto axis_dist(const IsPoint auto& a, const IsPoint auto& b, int axis) const {
    return std::abs(a.coords[axis] - b.coords[axis]);
  }
};

struct L2 {
  double dist(const IsPoint auto& a, const IsPoint auto& b) const {
    return std::hypot(static_cast<double>(a.x) - b.x, static_cast<double>(a.y) - b.y);
  }
  double axis_dist(const IsPoint auto& a, const IsPoint auto& b, int axis) const {
    return std::abs(static_cast<double>(a.coords[axis]) - b.coords[axis]);
  }
};

struct L2sq {
  auto dist(const IsPoint auto& a, const IsPoint auto& b) const {
    auto dx = a.x - b.x;
    auto dy = a.y - b.y;
    return dx * dx + dy * dy;
  }
  auto axis_dist(const IsPoint auto& a, const IsPoint auto& b, int axis) const {
    auto d = a.coords[axis] - b.coords[axis];
    return d * d;
  }
};

struct Linf {
  auto dist(const IsPoint auto& a, const IsPoint auto& b) const {
    return std::max(std::abs(a.x - b.x), std::abs(a.y - b.y));
  }
  auto axis_dist(const IsPoint auto& a, const IsPoint auto& b, int axis) const {
    return std::abs(a.coords[axis] - b.coords[axis]);
  }
};

template <typename BaseMetric, typename T>
struct Toroidal {
  Point<T> bounds;
  BaseMetric base;

  Toroidal(Point<T> b) : bounds(b), base() {}

  auto dist(const IsPoint auto& a, const IsPoint auto& b) const {
    auto dx = std::abs(a.x - b.x);
    auto dy = std::abs(a.y - b.y);
    dx = std::min(dx, static_cast<decltype(dx)>(bounds.x) - dx);
    dy = std::min(dy, static_cast<decltype(dy)>(bounds.y) - dy);
    return base.dist(Point<decltype(dx)>(0, 0), Point<decltype(dx)>(dx, dy));
  }

  auto axis_dist(const IsPoint auto& a, const IsPoint auto& b, int axis) const {
    auto d = std::abs(a.coords[axis] - b.coords[axis]);
    auto bd = static_cast<decltype(d)>(bounds.coords[axis]);
    auto dist = std::min(d, bd - d);
    Point<decltype(dist)> p1(0, 0), p2(0, 0);
    p2.coords[axis] = dist;
    return base.axis_dist(p1, p2, axis);
  }
};

struct GreatCircle {
  double radius = 6371000.0; // Earth radius in meters

  double dist(const IsPoint auto& a, const IsPoint auto& b) const {
    const double PI = 3.14159265358979323846;
    double lat1 = static_cast<double>(a.x) * PI / 180.0;
    double lat2 = static_cast<double>(b.x) * PI / 180.0;
    double dlat = static_cast<double>(b.x - a.x) * PI / 180.0;
    double dlon = static_cast<double>(b.y - a.y) * PI / 180.0;

    double sa = std::sin(dlat / 2);
    double sb = std::sin(dlon / 2);
    double x = sa * sa + std::cos(lat1) * std::cos(lat2) * sb * sb;
    return 2.0 * radius * std::asin(std::sqrt(x));
  }

  double axis_dist(const IsPoint auto& a, const IsPoint auto& b, int axis) const {
    const double PI = 3.14159265358979323846;
    if (axis == 0) return static_cast<double>(std::abs(a.x - b.x)) * (PI * radius / 180.0);
    return 0.0; // Disable pruning on longitude for absolute correctness in this simple version
  }
};

template<IsPoint PointType = Pointd, class ValueType = int64_t>
class KDTree {
 public:
  using point_type = PointType;
  using value_type = ValueType;
  using Entry = kdtree::Entry<PointType, ValueType>;

 private:
  struct Node {
    Entry entry;
    int depth;
    std::unique_ptr<Node> children[2];

    Node(Entry e, int d) : entry(e), depth(d) {}
  };

 public:
  class Iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = Entry;
    using pointer = Entry*;
    using reference = Entry&;
    using difference_type = std::ptrdiff_t;

    Iterator() {}
    Iterator(const Node* n) { if (n) stack.push_back(n); }

    const Entry& operator*() const {
      assert(!stack.empty());
      return stack.back()->entry;
    }
    const Entry* operator->() const {
      assert(!stack.empty());
      return &stack.back()->entry;
    }
    Iterator& operator++() {
      assert(!stack.empty());
      const Node* node = stack.back();
      stack.pop_back();
      if (node->children[1]) {
        stack.push_back(node->children[1].get());
      }
      if (node->children[0]) {
        stack.push_back(node->children[0].get());
      }
      return *this;
    }
    Iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }
    bool operator==(const Iterator&) const = default;
    bool operator!=(const Iterator&) const = default;
   private:
    std::vector<const Node*> stack;
  };
  using const_iterator = Iterator;

  KDTree() : root(nullptr), count(0), sum_depth(0) {}
  KDTree(const std::vector<Entry>& entries) : KDTree() {
    std::vector<std::unique_ptr<Node>> nodes;
    nodes.reserve(entries.size());
    for (const auto& e : entries) {
      nodes.push_back(std::make_unique<Node>(e, 0));
    }
    root = build_balanced_tree(nodes.begin(), nodes.end(), 0);
    assert(entries.size() == size());
  }

  bool empty() const { return root == nullptr; }
  size_t size() const { return static_cast<size_t>(count); }
  void clear() {
    root.reset();
    count = 0;
    sum_depth = 0;
  }

  Iterator begin() const { return Iterator(root.get()); }
  Iterator end() const { return Iterator(); }

  bool insert(PointType p, ValueType v) { return insert({p, v}); }
  bool insert(Entry e) { return insert_or_set(e, false); }
  bool set(PointType p, ValueType v) { return set({p, v}); }
  bool set(Entry e) { return insert_or_set(e, true); }

  bool remove(PointType p) {
    std::unique_ptr<Node>* node = &root;
    while (*node) {
      if ((*node)->entry.p == p) {
        remove_node(*node);
        return true;
      }
      int axis = (*node)->depth % 2;
      int child = (p.coords[axis] < (*node)->entry.p.coords[axis] ? 0 : 1);
      node = &(*node)->children[child];
    }
    return false;
  }

  bool exists(PointType p) const {
    return bool(find(p));
  }

  std::optional<Entry> find(PointType p) const {
    Node* node = root.get();
    while (node) {
      if (node->entry.p == p) {
        return node->entry;
      }
      int axis = node->depth % 2;
      int child = (p.coords[axis] < node->entry.p.coords[axis] ? 0 : 1);
      node = node->children[child].get();
    }
    return {};
  }

  template <typename Metric = L2sq>
    requires IsMetric<Metric, PointType, PointType>
  Entry find_closest(const IsPoint auto& p, const Metric& metric = Metric{}) const {
    assert(root != nullptr);
    return *find_closest(p, metric, -1);
  }

  template <typename Metric>
  std::optional<Entry> find_closest(const IsPoint auto& p, const Metric& metric, double max_dist) const
    requires IsMetric<Metric, decltype(p), PointType>
  {
    if (!root) {
      return std::nullopt;
    }

    using DistanceType = decltype(metric.dist(p, root->entry.p));
    const Node* best_node = nullptr;
    DistanceType best_dist = (max_dist < 0 ? std::numeric_limits<DistanceType>::max()
                                           : static_cast<DistanceType>(max_dist));

    find_closest_impl<Metric>(root.get(), p, best_dist, best_node, metric);

    if (best_node) {
      return best_node->entry;
    }
    return std::nullopt;
  }

  template <typename Metric = L2sq>
  std::vector<Entry> find_closest_k(const IsPoint auto& p, size_t k, const Metric& metric = Metric{}, double max_dist = -1.0) const
    requires IsMetric<Metric, decltype(p), PointType>
  {
    if (!root || k == 0) {
      return {};
    }

    using DistanceType = decltype(metric.dist(p, root->entry.p));
    std::priority_queue<std::pair<DistanceType, const Node*>> pq;
    DistanceType best_dist = (max_dist < 0 ? std::numeric_limits<DistanceType>::max()
                                           : static_cast<DistanceType>(max_dist));

    find_closest_k_impl<Metric>(root.get(), p, k, best_dist, pq, metric);

    std::vector<Entry> results;
    results.reserve(pq.size());
    while (!pq.empty()) {
      results.push_back(pq.top().second->entry);
      pq.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
  }

  template <typename Metric>
  std::vector<Entry> find_all_within(const IsPoint auto& p, const Metric& metric, double radius) const
    requires IsMetric<Metric, decltype(p), PointType>
  {
    if (!root) {
      return {};
    }
    std::vector<Entry> results;
    using DistanceType = decltype(metric.dist(p, root->entry.p));
    find_all_within_impl<Metric>(root.get(), p, static_cast<DistanceType>(radius), results, metric);
    return results;
  }


  template <typename Metric = L2sq>
  Entry pop_closest(const IsPoint auto& p, const Metric& metric = Metric{})
    requires IsMetric<Metric, decltype(p), PointType>
  {
    assert(root != nullptr);
    return *pop_closest(p, metric, -1);
  }

  template <typename Metric>
  std::optional<Entry> pop_closest(const IsPoint auto& p, const Metric& metric, double max_dist)
    requires IsMetric<Metric, decltype(p), PointType>
  {
    if (!root) {
      return std::nullopt;
    }

    const Node* best_node = nullptr;

    using DistanceType = decltype(metric.dist(p, root->entry.p));
    DistanceType best_dist = (max_dist < 0 ? std::numeric_limits<DistanceType>::max()
                                           : static_cast<DistanceType>(max_dist));

    find_closest_impl<Metric>(root.get(), p, best_dist, best_node, metric);

    if (best_node) {
      Entry out = best_node->entry;
      remove(best_node->entry.p);
      return out;
    }
    return std::nullopt;
  }

  void print_tree(std::ostream& stream = std::cout) const {
    if (root) {
      stream << root->entry << std::endl;
      print_tree(stream, root->children[0].get(), "", true);
      print_tree(stream, root->children[1].get(), "", false);
    }
  }

  friend std::ostream& operator<<(std::ostream& stream, const KDTree& t) {
    t.print_tree(stream);
    return stream;
  }

  void validate() const;  // Implemented in test file

  void rebalance() {
    if (!root) {
      return;
    }

    std::vector<std::unique_ptr<Node>> nodes;
    nodes.reserve(count);
    collect_nodes(std::move(root), nodes);

    assert(!root);
    assert(sum_depth == 0);
    assert(count == 0);

    root = build_balanced_tree(nodes.begin(), nodes.end(), 0);
    assert(nodes.size() == size());
  }

  std::string balance_str() const {
    return std::format("size: {}, max depth: {}, avg depth: {:.3f}, std dev: {:.3f}, balance: {:.3f}",
                       size(), depth_max(), depth_avg(), depth_stddev(), balance_factor());
  }

  size_t depth_max() const {
    return static_cast<size_t>(depth_max(root.get()));
  }

  double depth_avg() const {
    if (count > 0) {
      return static_cast<double>(sum_depth) / count;
    } else {
      return 0;
    }
  }

  double depth_stddev() const {
    if (count > 0) {
      auto [_, variance] = depth_variance(root.get());
      return std::sqrt(variance / count);
    } else {
      return 0;
    }
  }

  double balance_factor() const {
    if (count == 0) {
      return 1;
    } else {
      int leaves = leaf_count(root.get());
      return 2.0 * leaves / count;
    }
  }

 private:
  std::unique_ptr<Node> root;
  int count;
  int64_t sum_depth;

  bool insert_or_set(Entry e, bool replace) {
    std::unique_ptr<Node>* node = &root;
    int depth = 0;
    while (*node) {
      if ((*node)->entry.p == e.p) {
        if (replace) {
          (*node)->entry.value = e.value;
        }
        return false; // Point already exists
      }
      int axis = depth % 2;
      int child = (e.p.coords[axis] < (*node)->entry.p.coords[axis] ? 0 : 1);
      node = &(*node)->children[child];
      depth += 1;
    }
    *node = std::make_unique<Node>(e, depth);
    count += 1;
    sum_depth += depth;

    if (sum_depth > std::bit_width(static_cast<unsigned>(count)) * count + 1) {
      // `bit_width(count)` is the max depth for a complete balanced tree.
      // `sum_depth / count` is the average depth, which should be 1-2 less than the
      // max for a balanced tree, but will exceed that if it's sufficiently unbalanced.
      // Move `/ count` to the other side to avoid division, in particular by 0.
      // There may be better metrics for how balanced a tree is, but this one is cheap
      // and easy to compute incrementally and seems to work.
      rebalance();
    }

    return true;
  }

  // Const version for find_closest
  template <typename Metric, typename DistanceType>
  void find_closest_impl(
      const Node* node, const IsPoint auto& p,
      DistanceType& best_dist,
      const Node*& best_node,
      const Metric& metric) const {
    if (!node) {
      return;
    }

    auto dist = metric.dist(node->entry.p, p);
    if (dist <= best_dist) {
      best_dist = dist;
      best_node = node;
    }

    int axis = node->depth % 2;
    int search_first = (p.coords[axis] < node->entry.p.coords[axis]) ? 0 : 1;
    find_closest_impl<Metric>(node->children[search_first].get(), p, best_dist, best_node, metric);

    auto ad = metric.axis_dist(p, node->entry.p, axis);
    if (ad <= best_dist) {
      find_closest_impl<Metric>(node->children[!search_first].get(), p, best_dist, best_node, metric);
    }
  }

  // Mutable version for pop_closest
  template <typename Metric, typename DistanceType>
  void find_closest_impl_mutable(
      std::unique_ptr<Node>* node, const IsPoint auto& p,
      DistanceType& best_dist,
      std::unique_ptr<Node>*& best_node,
      const Metric& metric) {
    if (!*node) {
      return;
    }

    auto dist = metric.dist((*node)->entry.p, p);
    if (dist <= best_dist) {
      best_dist = dist;
      best_node = node;
    }

    int axis = (*node)->depth % 2;
    int search_first = (p.coords[axis] < (*node)->entry.p.coords[axis]) ? 0 : 1;
    find_closest_impl_mutable<Metric>(&(*node)->children[search_first], p, best_dist, best_node, metric);

    auto ad = metric.axis_dist(p, (*node)->entry.p, axis);
    if (ad <= best_dist) {
      find_closest_impl_mutable<Metric>(&(*node)->children[!search_first], p, best_dist, best_node, metric);
    }
  }

  template <typename Metric, typename DistanceType>
  void find_closest_k_impl(
      const Node* node, const IsPoint auto& p, size_t k,
      DistanceType& best_dist,
      std::priority_queue<std::pair<DistanceType, const Node*>>& pq,
      const Metric& metric) const {
    if (!node) {
      return;
    }

    auto dist = metric.dist(node->entry.p, p);
    if (dist <= best_dist) {
      pq.push({dist, node});
      if (pq.size() > k) {
        pq.pop();
      }
      if (pq.size() == k) {
        best_dist = pq.top().first;
      }
    }

    int axis = node->depth % 2;
    int search_first = (p.coords[axis] < node->entry.p.coords[axis]) ? 0 : 1;
    find_closest_k_impl<Metric>(node->children[search_first].get(), p, k, best_dist, pq, metric);

    auto ad = metric.axis_dist(p, node->entry.p, axis);
    if (ad <= best_dist) {
      find_closest_k_impl<Metric>(node->children[!search_first].get(), p, k, best_dist, pq, metric);
    }
  }

  template <typename Metric, typename DistanceType>
  void find_all_within_impl(
      const Node* node, const IsPoint auto& p,
      DistanceType radius,
      std::vector<Entry>& results,
      const Metric& metric) const {
    if (!node) {
      return;
    }

    if (metric.dist(node->entry.p, p) <= radius) {
      results.push_back(node->entry);
    }

    int axis = node->depth % 2;
    int search_first = (p.coords[axis] < node->entry.p.coords[axis]) ? 0 : 1;
    find_all_within_impl<Metric>(node->children[search_first].get(), p, radius, results, metric);

    auto ad = metric.axis_dist(p, node->entry.p, axis);
    if (ad <= radius) {
      find_all_within_impl<Metric>(node->children[!search_first].get(), p, radius, results, metric);
    }
  }

  void find_leftmost_along_axis(
      std::unique_ptr<Node>* node, typename PointType::value_type coord, int axis,
      typename PointType::value_type& best_dist,
      std::unique_ptr<Node>*& best_node) const {
    if (!*node) {
      return;
    }

    typename PointType::value_type dist = (*node)->entry.p.coords[axis] - coord;
    if (dist < best_dist ||
        (dist == best_dist && (!best_node || (*node)->depth > (*best_node)->depth))) {
      // Prioritizing the deepest means less cascading of intermediate nodes being replaced
      // by deeper nodes, or rebuilding a smaller tree.
      best_dist = dist;
      best_node = node;
    }

    find_leftmost_along_axis(&(*node)->children[0], coord, axis, best_dist, best_node);
    if (axis != (*node)->depth % 2) {
      // No need to search the right side if we're searching along the axis as they have
      // values greater or equal than this node. It's plausible there's a deeper node with
      // equal value, but it's probably not worth the effort to search for. We do need to
      // search the right side for off-axis levels as we make no claim about them.
      find_leftmost_along_axis(&(*node)->children[1], coord, axis, best_dist, best_node);
    }
  }

  void remove_node(std::unique_ptr<Node>& node) {
    if (!node) {
      return;
    } else if (node->children[1]) {
      // It is valid to replace this node with any of the leftmost nodes in the right subtree.
      // There may be multiple leftmost nodes, but any will do as they will all sort to the
      // right of any of the others.
      std::unique_ptr<Node>* best_node = nullptr;
      typename PointType::value_type best_dist = std::numeric_limits<typename PointType::value_type>::max();
      int axis = node->depth % 2;
      typename PointType::value_type coord = node->entry.p.coords[axis];
      find_leftmost_along_axis(&node->children[1], coord, axis, best_dist, best_node);

      // If there is a right subtree, there will be a left-most node with value >= this one.
      assert(best_dist >= 0);
      assert(best_node);

      node->entry = (*best_node)->entry;
      remove_node(*best_node);
      return;
    } else if (node->children[0]) {
      // It is NOT valid to replace this node with the rightmost node of the left subtree,
      // as promoting that node would break the invariant for all nodes that have a value
      // equal to it along that axis, so they'd need to move from the left subtree to the
      // right subtree. Finding/moving all of those would be a pain, so just rebuild instead.
      int depth = node->depth;
      sum_depth -= depth;
      count -= 1;
      std::vector<std::unique_ptr<Node>> nodes;
      // Skip collecting this node as it's being removed.
      collect_nodes(std::move(node->children[0]), nodes);
      assert(!node->children[1]);  // Otherwise we'd have replaced this node above.
      node = build_balanced_tree(nodes.begin(), nodes.end(), depth);
      return;
    } else {
      // A leaf node can just be removed.
      sum_depth -= node->depth;
      count -= 1;
      node.reset();
      return;
    }
  }

  void print_tree(std::ostream& stream, const Node* node, std::string prefix, bool first) const {
    if (!node) {
      return;
    }
    stream << prefix << (first ? "├─" : "└─") << node->entry << std::endl;
    prefix += (first ? "│ " : "  ");
    print_tree(stream, node->children[0].get(), prefix, true);
    print_tree(stream, node->children[1].get(), prefix, false);
  }

  int64_t validate(const Node* node, int depth, PointType min, PointType max) const;

  int depth_max(const Node* node) const {
    if (!node) {
      return 0;
    }
    return std::max({
      node->depth,
      depth_max(node->children[0].get()),
      depth_max(node->children[1].get())
    });
  }

  int leaf_count(const Node* node) const {
    if (!node) {
      return 0;
    } else if (!node->children[0] && !node->children[1]) {
      return 1;
    } else {
      return (leaf_count(node->children[0].get()) +
              leaf_count(node->children[1].get()));
    }
  }

  std::pair<int, double> depth_variance(Node* node) const {
    if (!node) return {0, 0.0};

    auto [left_height, left_variance] = depth_variance(node->children[0].get());
    auto [right_height, right_variance] = depth_variance(node->children[1].get());

    int height = std::max(left_height, right_height) + 1;
    int height_diff = left_height - right_height;
    double variance = left_variance + right_variance + height_diff * height_diff;
    return {height, variance};
  }

  void collect_nodes(std::unique_ptr<Node> node, std::vector<std::unique_ptr<Node>>& nodes) {
    if (!node) {
      return;
    }
    collect_nodes(std::move(node->children[0]), nodes);
    collect_nodes(std::move(node->children[1]), nodes);
    sum_depth -= node->depth;
    count -= 1;
    nodes.push_back(std::move(node));
  }

  std::unique_ptr<Node> build_balanced_tree(
      typename std::vector<std::unique_ptr<Node>>::iterator start,
      typename std::vector<std::unique_ptr<Node>>::iterator end,
      int depth) {
    if (start == end) {
      return nullptr;
    }

    int axis = depth % 2;

    // Choose the pivot.
    auto mid = std::next(start, std::distance(start, end) / 2);
    // Find the pivot value
    std::nth_element(start, mid, end, [axis](const std::unique_ptr<Node>& a, const std::unique_ptr<Node>& b) {
      return a->entry.p.coords[axis] < b->entry.p.coords[axis];
    });

    typename PointType::value_type pivot_coord = (*mid)->entry.p.coords[axis];
    // Find the pivot's true location, as it has to be the first of that value.
    mid = std::partition(start, mid, [axis, pivot_coord](const std::unique_ptr<Node>& a) {
      return a->entry.p.coords[axis] < pivot_coord;
    });

    std::unique_ptr<Node> node = std::move(*mid);
    node->depth = depth;
    sum_depth += depth;
    count += 1;
    node->children[0] = build_balanced_tree(start, mid, depth + 1);
    node->children[1] = build_balanced_tree(mid + 1, end, depth + 1);

    return node;
  }
};

using KDTreei = KDTree<Pointi, int64_t>;
using KDTreef = KDTree<Pointf, int64_t>;
using KDTreed = KDTree<Pointd, int64_t>;

}  // namespace kdtree
