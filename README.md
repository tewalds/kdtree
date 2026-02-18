# KDTree: Dynamic 2D Spatial Index

A high-performance, header-only C++ KD-tree implementation with Python bindings. Unlike most KD-tree implementations, this one supports **dynamic insertion and removal** without rebuilding the entire tree.

## Why This KDTree?

| Feature | This KDTree | scipy.spatial.KDTree |
|---------|-------------|---------------------|
| Insert after build | ✅ O(log n) | ❌ Requires full rebuild |
| Remove points | ✅ O(log n) | ❌ Requires full rebuild |
| Nearest neighbor | ✅ O(log n) | ✅ O(log n) |
| Auto-rebalancing | ✅ Automatic | ❌ Manual rebuild |
| Distance metrics | L1, L2, Linf (runtime) | L1, L2, Lp, custom |
| Language | C++20 / Python | Python |
| K dimensions | 2D only | Any K |

**Perfect for:** Games, simulations, real-time systems, or any application where spatial data changes frequently.

**Not for:** High-dimensional data (use Annoy, FAISS) or static datasets (scipy is fine).

## Quick Start

### C++

```cpp
#include "kdtree.h"
using namespace kdtree;

KDTreed tree;  // Pointd coords, int64_t values
tree.insert(Pointd(1.5, 2.3), 42);
tree.insert(Pointd(4.1, 3.7), 7);

auto result = tree.find_closest(Pointd(2.0, 3.0));
std::cout << "Closest: " << result << std::endl; // Entry({1.5, 2.3}, 42)

// Iterate with structured bindings
for (auto [p, v] : tree) {
  std::cout << "Point: " << p << ", Value: " << v << std::endl;
}

// Manhattan distance
auto manhattan = tree.find_closest(Pointd(2.0, 3.0), Norm::L1);
```

**Build:**
```bash
mkdir build && cd build
cmake ..
make
./kdtree_test  # Run tests
```

### Python

```bash
pip install .
```

```python
import kdtree

tree = kdtree.KDTreed()
tree.insert((1.5, 2.3), 42)
tree.insert(4.1, 3.7, 7)

result = tree.find_closest((2.0, 3.0))
print(f"Closest: {result}") # Entry({1.5, 2.3}, 42)

# Manhattan distance
result_l1 = tree.find_closest((2.0, 3.0), kdtree.Norm.L1)
```

## Features

### Dynamic Updates

```python
# Bad: scipy approach - rebuild entire tree on every change
entities_moved = True
if entities_moved:
    kdtree = scipy.spatial.KDTree(positions)  # O(n log n) rebuild!

# Good: This KDTree - incremental updates
for entity in moved_entities:
    tree.remove(entity.old_pos)         # O(log n)
    tree.insert(entity.new_pos, entity.id) # O(log n)
```

### Distance Metrics

L1 (Manhattan), L2 (Euclidean squared), and Linf (Chebyshev) are supported via runtime parameter:

```python
tree = kdtree.KDTreed()
# ... insert points ...

# Euclidean (default)
closest = tree.find_closest((1.0, 2.0))
closest = tree.find_closest((1.0, 2.0), kdtree.Norm.L2)

# Manhattan - same tree, different query
closest = tree.find_closest((1.0, 2.0), kdtree.Norm.L1)

# Chebyshev
closest = tree.find_closest((1.0, 2.0), kdtree.Norm.Linf)
```

**Note:** The tree structure is independent of the distance metric, so you can use different queries on the same tree!

### Python Object Storage

Store arbitrary Python objects directly (no index indirection needed):

```python
tree = kdtree.KDTreePyd()  # "Py" = Python object storage
tree.insert((1.0, 2.0), {"name": "Alice", "data": [1,2,3]})
tree.insert((3.0, 4.0), MyCustomClass())

result = tree.find_closest((2.0, 3.0))
obj = result.value  # Original Python object
```

### Automatic Rebalancing

The tree automatically rebalances when it detects imbalance:

```python
tree = kdtree.KDTreed()
for i in range(10000):
    tree.insert((random(), random()), i)
# Automatically rebalances as needed - no manual intervention

# Or force rebalance:
tree.rebalance()
print(tree.balance_str())
# Output: size: 10000, max depth: 18, avg depth: 12.4, ...
```

## API Reference

### C++

```cpp
namespace kdtree {
  // Point types
  using Pointi = Point<int>;
  using Pointf = Point<float>;
  using Pointd = Point<double>;

  // Entry container (lives outside KDTree class)
  template<typename PointType, typename ValueType>
  struct Entry { PointType p; ValueType value; };

  // Tree types (all use int64_t values by default)
  using KDTreei = KDTree<Pointi, int64_t>;
  using KDTreef = KDTree<Pointf, int64_t>;
  using KDTreed = KDTree<Pointd, int64_t>;  // Recommended

  // Or use custom types:
  KDTree<Pointd, MyStruct> tree;  // Any copyable type works
}
```

**Methods:**
- `bool insert(PointType p, ValueType v)` - Insert key/value, returns false if point exists
- `bool insert(Entry e)` - Insert entry
- `bool set(PointType p, ValueType v)` - Set key/value, returns false if point exists
- `bool set(Entry e)` - Set entry
- `bool remove(PointType p)` - Remove by point
- `bool exists(PointType p)` - Check existence
- `optional<Entry> find(PointType p)` - Exact lookup
- `Entry find_closest(PointType p, Norm norm=L2)` - Nearest neighbor
- `Entry pop_closest(PointType p, Norm norm=L2)` - Remove & return nearest
- `void rebalance()` - Force rebalance
- `size_t size()`, `bool empty()`, `void clear()`
- `Iterator begin/end()` - Range-based for loop support

### Python

**Tree Types (4 total):**
```python
# int64_t storage (for indices/IDs)
kdtree.KDTreei   # int coords
kdtree.KDTreed   # double coords (recommended)

# Python object storage (any object)
kdtree.KDTreePyi  # int coords
kdtree.KDTreePyd  # double coords
```

**Insert Convenience:**
```python
tree.insert(point, value)          # Point, Value type
tree.insert((1.0, 2.0), 42)        # tuple, value
tree.insert([1.0, 2.0], 42)        # list, value
tree.insert(1.0, 2.0, 42)          # x, y, value
```

**Query Convenience:**
```python
tree.find_closest((1.0, 2.0))           # tuple
tree.find_closest(1.0, 2.0)             # x, y
tree.find_closest(point, kdtree.Norm.L1)  # with norm parameter
```

## Performance

- **Insert/Remove:** O(log n) average, O(n) worst case (triggers rebalance)
- **Find/Exists:** O(log n) average
- **Nearest neighbor:** O(log n) average, O(n) worst case
- **Rebalance:** O(n log n)
- **Memory:** ~40-48 bytes per node depending on coordinate type

## Requirements

- **C++:** C++20 compiler (GCC 10+, Clang 12+, MSVC 2019+)
- **Python:** Python 3.8+, pybind11

## Installation

### C++ (header-only)

```bash
# Copy kdtree.h to your include path
```

### Python

```bash
pip install .
# or for development:
pip install -e .
```

## Limitations

- **2D only** - Not suitable for 3D+ data
- **Not thread-safe** - Use external synchronization
- **No k-nearest** - Only single nearest neighbor (for now)

For k-dimensional trees or k-nearest neighbors, see scipy, Annoy, or nanoflann.

## License

Apache License 2.0 - See LICENSE file

## Contributing

Contributions welcome! Areas of interest:
- [ ] k-nearest neighbors (not just nearest)
- [ ] Range queries (all points within radius)
- [ ] Benchmarks vs scipy
- [ ] 3D support (relatively straightforward)

## See Also

- [scipy.spatial.KDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) - Static KD-trees, k-dimensional
- [nanoflann](https://github.com/jlblancoc/nanoflann) - Header-only C++ KD-tree, k-dimensional
- [Annoy](https://github.com/spotify/annoy) - Approximate nearest neighbors, high-dimensional