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
#include <string>
#include <utility>
#include <vector>

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

  double distance(const Point& o) const {
    return std::hypot(static_cast<double>(x - o.x), static_cast<double>(y - o.y));
  }

  friend std::ostream& operator<<(std::ostream& stream, const Point& p) {
    return stream << "{" << p.x << ", " << p.y << "}";
  }
};

using Pointi = Point<int>;
using Pointf = Point<float>;
using Pointd = Point<double>;

template <typename ValueType, typename PointType>
struct Value {
  ValueType value;
  PointType p;

  Value() = default;
  Value(ValueType v, PointType pt) : value(v), p(pt) {}

  bool operator==(const Value& o) const = default;
  bool operator!=(const Value& o) const = default;

  friend std::ostream& operator<<(std::ostream& stream, const Value& v) {
    return stream << "Value(" << v.value << ", " << v.p << ")";
  }
};

enum class Norm { L1, L2, Linf };

template<class ValueType = int64_t, class PointType = Pointd>
class KDTree {
 public:
  using value_type = ValueType;
  using point_type = PointType;
  using Value = kdtree::Value<ValueType, PointType>;

 private:
  struct Node {
    Value value;
    int depth;
    std::unique_ptr<Node> children[2];

    Node(Value v, int d) : value(v), depth(d) {}
  };

 public:
  class Iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = Value;
    using pointer = Value*;
    using reference = Value&;
    using difference_type = std::ptrdiff_t;

    Iterator() {}
    Iterator(const Node* n) { if (n) stack.push_back(n); }

    const Value& operator*() const {
      assert(!stack.empty());
      return stack.back()->value;
    }
    const Value* operator->() const {
      assert(!stack.empty());
      return &stack.back()->value;
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
  KDTree(const std::vector<Value>& values) : KDTree() {
    std::vector<Value> mut_values = values;  // Copy
    root = build_balanced_tree(mut_values.begin(), mut_values.end(), 0);
    assert(values.size() == size());
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

  bool insert(ValueType v, PointType p) { return insert({v, p}); }
  bool insert(Value v) { return insert_or_set(v, false); }
  bool set(ValueType v, PointType p) { return set({v, p}); }
  bool set(Value v) { return insert_or_set(v, true); }

  bool remove(PointType p) {
    std::unique_ptr<Node>* node = &root;
    while (*node) {
      if ((*node)->value.p == p) {
        remove_node(*node);
        return true;
      }
      int axis = (*node)->depth % 2;
      int child = (p.coords[axis] < (*node)->value.p.coords[axis] ? 0 : 1);
      node = &(*node)->children[child];
    }
    return false;
  }

  bool exists(PointType p) const {
    return bool(find(p));
  }

  std::optional<Value> find(PointType p) const {
    Node* node = root.get();
    while (node) {
      if (node->value.p == p) {
        return node->value;
      }
      int axis = node->depth % 2;
      int child = (p.coords[axis] < node->value.p.coords[axis] ? 0 : 1);
      node = node->children[child].get();
    }
    return {};
  }

  Value find_closest(PointType p, Norm norm = Norm::L2) const {
    assert(root != nullptr);
    const std::unique_ptr<Node>* best_node = nullptr;
    typename PointType::value_type best_dist = std::numeric_limits<typename PointType::value_type>::max();

    switch(norm) {
      case Norm::L1: find_closest_impl<&distance_l1>(&root, p, best_dist, best_node); break;
      case Norm::L2: find_closest_impl<&distance_l2>(&root, p, best_dist, best_node); break;
      case Norm::Linf: find_closest_impl<&distance_linf>(&root, p, best_dist, best_node); break;
      // No default so it's a compilation error if they're out of sync.
    }

    assert(best_node);
    return (*best_node)->value;
  }

  Value pop_closest(PointType p, Norm norm = Norm::L2) {
    assert(root != nullptr);
    std::unique_ptr<Node>* best_node = nullptr;
    typename PointType::value_type best_dist = std::numeric_limits<typename PointType::value_type>::max();

    switch(norm) {
      case Norm::L1: find_closest_impl_mutable<&distance_l1>(&root, p, best_dist, best_node); break;
      case Norm::L2: find_closest_impl_mutable<&distance_l2>(&root, p, best_dist, best_node); break;
      case Norm::Linf: find_closest_impl_mutable<&distance_linf>(&root, p, best_dist, best_node); break;
    }

    assert(best_node);
    Value out = (*best_node)->value;
    remove_node(*best_node);
    return out;
  }

  void print_tree(std::ostream& stream = std::cout) const {
    if (root) {
      stream << root->value << std::endl;
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

    std::vector<Value> values;
    values.reserve(count);
    collect_values(root, values);

    assert(!root);
    assert(sum_depth == 0);
    assert(count == 0);

    root = build_balanced_tree(values.begin(), values.end(), 0);
    assert(values.size() == size());
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

  bool insert_or_set(Value v, bool replace) {
    std::unique_ptr<Node>* node = &root;
    int depth = 0;
    while (*node) {
      if ((*node)->value.p == v.p) {
        if (replace) {
          (*node)->value.value = v.value;
        }
        return false; // Value already exists
      }
      int axis = depth % 2;
      int child = (v.p.coords[axis] < (*node)->value.p.coords[axis] ? 0 : 1);
      node = &(*node)->children[child];
      depth += 1;
    }
    *node = std::make_unique<Node>(v, depth);
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

  static typename PointType::value_type distance_l1(PointType a, PointType b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
  }

  static typename PointType::value_type distance_l2(PointType a, PointType b) {
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
  }

  static typename PointType::value_type distance_linf(PointType a, PointType b) {
    return std::max(std::abs(a.x - b.x), std::abs(a.y - b.y));
  }

  // Const version for find_closest
  template<typename PointType::value_type (*DistFn)(PointType, PointType)>
  void find_closest_impl(
      const std::unique_ptr<Node>* node, PointType p,
      typename PointType::value_type& best_dist,
      const std::unique_ptr<Node>*& best_node) const {
    if (!*node) {
      return;
    }

    typename PointType::value_type dist = DistFn((*node)->value.p, p);
    if (dist < best_dist) {
      best_dist = dist;
      best_node = node;
    }

    int axis = (*node)->depth % 2;
    int search_first = (p.coords[axis] < (*node)->value.p.coords[axis]) ? 0 : 1;
    find_closest_impl<DistFn>(&(*node)->children[search_first], p, best_dist, best_node);

    typename PointType::value_type axis_dist = std::abs(p.coords[axis] - (*node)->value.p.coords[axis]);
    if (axis_dist < best_dist) {
      find_closest_impl<DistFn>(&(*node)->children[!search_first], p, best_dist, best_node);
    }
  }

  // Mutable version for pop_closest
  template<typename PointType::value_type (*DistFn)(PointType, PointType)>
  void find_closest_impl_mutable(
      std::unique_ptr<Node>* node, PointType p,
      typename PointType::value_type& best_dist,
      std::unique_ptr<Node>*& best_node) {
    if (!*node) {
      return;
    }

    typename PointType::value_type dist = DistFn((*node)->value.p, p);
    if (dist < best_dist) {
      best_dist = dist;
      best_node = node;
    }

    int axis = (*node)->depth % 2;
    int search_first = (p.coords[axis] < (*node)->value.p.coords[axis]) ? 0 : 1;
    find_closest_impl_mutable<DistFn>(&(*node)->children[search_first], p, best_dist, best_node);

    typename PointType::value_type axis_dist = std::abs(p.coords[axis] - (*node)->value.p.coords[axis]);
    if (axis_dist < best_dist) {
      find_closest_impl_mutable<DistFn>(&(*node)->children[!search_first], p, best_dist, best_node);
    }
  }

  void find_leftmost_along_axis(
      std::unique_ptr<Node>* node, typename PointType::value_type coord, int axis,
      typename PointType::value_type& best_dist,
      std::unique_ptr<Node>*& best_node) const {
    if (!*node) {
      return;
    }

    typename PointType::value_type dist = (*node)->value.p.coords[axis] - coord;
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
      typename PointType::value_type coord = node->value.p.coords[axis];
      find_leftmost_along_axis(&node->children[1], coord, axis, best_dist, best_node);

      // If there is a right subtree, there will be a left-most node with value >= this one.
      assert(best_dist >= 0);
      assert(best_node);

      node->value = (*best_node)->value;
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
      std::vector<Value> values;
      // Skip collecting this node as it's being removed.
      collect_values(node->children[0], values);
      assert(!node->children[1]);  // Otherwise we'd have replaced this node above.
      node = build_balanced_tree(values.begin(), values.end(), depth);
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
    stream << prefix << (first ? "├─" : "└─") << node->value << std::endl;
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

  void collect_values(std::unique_ptr<Node>& node, std::vector<Value>& values) {
    if (!node) {
      return;
    }
    values.push_back(node->value);
    collect_values(node->children[0], values);
    collect_values(node->children[1], values);
    sum_depth -= node->depth;
    count -= 1;
    node.reset();
  }

  std::unique_ptr<Node> build_balanced_tree(
      typename std::vector<Value>::iterator start,
      typename std::vector<Value>::iterator end,
      int depth) {
    if (start == end) {
      return nullptr;
    }

    int axis = depth % 2;

    // Choose the pivot.
    auto mid = std::next(start, std::distance(start, end) / 2);
    // Find the pivot value
    std::nth_element(start, mid, end, [axis](const Value& a, const Value& b) {
      return a.p.coords[axis] < b.p.coords[axis];
    });
    Value pivot = *mid;
    // Find the pivot's true location, as it has to be the first of that value.
    mid = std::partition(start, mid, [axis, &pivot](const Value& a) {
      return a.p.coords[axis] < pivot.p.coords[axis];
    });

    auto node = std::make_unique<Node>(*mid, depth);
    sum_depth += depth;
    count += 1;
    node->children[0] = build_balanced_tree(start, mid, depth + 1);
    node->children[1] = build_balanced_tree(mid + 1, end, depth + 1);

    return node;
  }
};

// Common type aliases
using KDTreei = KDTree<int64_t, Pointi>;
using KDTreef = KDTree<int64_t, Pointf>;
using KDTreed = KDTree<int64_t, Pointd>;

}  // namespace kdtree
