# KDTree: Dynamic 2D Spatial Index

A high-performance, header-only C++ KD-tree implementation with Python bindings. Unlike most KD-tree implementations, this one supports **dynamic insertion and removal** without rebuilding the entire tree.

## Why This KDTree?

| Feature | This KDTree | scipy.spatial.KDTree |
|---------|-------------|---------------------|
| Insert after build | ✅ O(log n) | ❌ Requires full rebuild |
| Remove points | ✅ O(log n) | ❌ Requires full rebuild |
| Nearest neighbor | ✅ O(log n) | ✅ O(log n) |
| Auto-rebalancing | ✅ Automatic | ❌ Manual rebuild |
| Distance metrics | L1, L2 (runtime) | L1, L2, Lp, custom |
| Language | C++20 / Python | Python |
| K dimensions | 2D only | Any K |

**Perfect for:** Games, simulations, real-time systems, or any application where spatial data changes frequently.

**Not for:** High-dimensional data (use Annoy, FAISS) or static datasets (scipy is fine).

## Quick Start

### C++

```cpp
#include "kdtree.h"
using namespace kdtree;

KDTreed tree;  // double coords, int64_t values
tree.insert({42, Pointd(1.5, 2.3)});
tree.insert({7, Pointd(4.1, 3.7)});

auto result = tree.find_closest(Pointd(2.0, 3.0));
std::cout << "Closest: " << result << std::endl;

// Manhattan distance
auto manhattan = tree.find_closest(Pointd(2.0, 3.0), Norm::L1);
```

**Build:**
```bash
mkdir build && cd build
cmake ..
make
ctest  # Run tests
```

### Python

```bash
pip install .
```

```python
import kdtree

tree = kdtree.KDTreed()
tree.insert(42, (1.5, 2.3))  # No Value type needed!
tree.insert(7, (4.1, 3.7))

result = tree.find_closest((2.0, 3.0))
print(f"Closest: {result}")

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
    tree.remove(entity.old_pos)      # O(log n)
    tree.insert(entity.id, entity.new_pos)  # O(log n)
```

### Distance Metrics

Both L1 (Manhattan) and L2 (Euclidean squared) are supported via runtime parameter:

```python
tree = kdtree.KDTreed()
# ... insert points ...

# Euclidean (default)
closest = tree.find_closest((1.0, 2.0))

# Manhattan - same tree, different query
closest = tree.find_closest((1.0, 2.0), kdtree.Norm.L1)
```

**Note:** The tree structure is independent of the distance metric, so you can use both L1 and L2 queries on the same tree!

### Python Object Storage

Store arbitrary Python objects directly (no index indirection needed):

```python
tree = kdtree.KDTreePyd()  # "Py" = Python object storage
tree.insert({"name": "Alice", "data": [1,2,3]}, (1.0, 2.0))
tree.insert(MyCustomClass(), (3.0, 4.0))

result = tree.find_closest((2.0, 3.0))
obj = result.value  # Original Python object
```

### Automatic Rebalancing

The tree automatically rebalances when it detects imbalance:

```python
tree = kdtree.KDTreed()
for i in range(10000):
    tree.insert(i, (random(), random()))
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

  // Value container (lives outside KDTree class)
  template<typename ValueType, typename PointType>
  struct Value { ValueType value; PointType p; };

  // Tree types (all use int64_t values by default)
  using KDTreei = KDTree<int64_t, Pointi>;
  using KDTreef = KDTree<int64_t, Pointf>;
  using KDTreed = KDTree<int64_t, Pointd>;  // Recommended

  // Or use custom types:
  KDTree<MyStruct, Pointd> tree;  // Any copyable type works
}
```

**Methods:**
- `bool insert(Value v)` - Insert value, returns false if point exists
- `bool remove(PointType p)` - Remove by point
- `bool exists(PointType p)` - Check existence
- `optional<Value> find(PointType p)` - Exact lookup
- `Value find_closest(PointType p, Norm norm=L2)` - Nearest neighbor
- `Value pop_closest(PointType p, Norm norm=L2)` - Remove & return nearest
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

**Note:** No `KDTreef` / `KDTreePyf` in Python. Python `float` is 64-bit (C++ `double`), so using C++ `float` (32-bit) would silently truncate precision and break `exists()` / `remove()`.

**Insert Convenience - Never spell Value type:**
```python
tree.insert(value, point)          # Value type, Point
tree.insert(42, (1.0, 2.0))        # value, tuple
tree.insert(42, [1.0, 2.0])        # value, list
tree.insert(42, 1.0, 2.0)          # value, x, y
```

**Query Convenience - All methods accept multiple forms:**
```python
tree.find_closest((1.0, 2.0))           # tuple
tree.find_closest(1.0, 2.0)             # x, y
tree.find_closest(point, kdtree.Norm.L1)  # with norm parameter
```

All methods (`exists`, `find`, `remove`, `find_closest`, `pop_closest`) accept:
- Explicit `Point` object
- Tuple: `(x, y)`
- List: `[x, y]`
- Separate coords: `x, y`

## Performance

- **Insert/Remove:** O(log n) average, O(n) worst case (triggers rebalance)
- **Find/Exists:** O(log n) average
- **Nearest neighbor:** O(log n) average, O(n) worst case
- **Rebalance:** O(n log n)
- **Memory:** ~40-48 bytes per node depending on coordinate type

Memory breakdown:
- `KDTreei`: ~40 bytes/node (int coords: 8, int64 value: 8, depth: 4, pointers: 16, padding: 4)
- `KDTreed`: ~48 bytes/node (double coords: 16, int64 value: 8, depth: 4, pointers: 16, padding: 4)

## Requirements

- **C++:** C++20 compiler (GCC 10+, Clang 12+, MSVC 2019+)
- **Python:** Python 3.8+, pybind11

## Installation

### C++ (header-only)

```bash
# Copy kdtree.h to your include path, or:
cmake -B build
cmake --build build
cmake --install build
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
- **No persistence** - No serialization (yet)
- **No k-nearest** - Only single nearest neighbor (for now)

For k-dimensional trees or k-nearest neighbors, see scipy, Annoy, or nanoflann.

## License

Apache License 2.0 - See LICENSE file

## Contributing

Contributions welcome! Areas of interest:
- [ ] k-nearest neighbors (not just nearest)
- [ ] Range queries (all points within radius)
- [ ] Serialization support
- [ ] Benchmarks vs scipy
- [ ] 3D support (relatively straightforward)

## See Also

- [scipy.spatial.KDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) - Static KD-trees, k-dimensional
- [nanoflann](https://github.com/jlblancoc/nanoflann) - Header-only C++ KD-tree, k-dimensional
- [Annoy](https://github.com/spotify/annoy) - Approximate nearest neighbors, high-dimensional