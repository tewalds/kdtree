# KDTree TODO & Improvements

## Search Enhancements
- [x] `find_closest(point, max_dist=inf, norm=L2)`: Return nearest neighbor only if within `max_dist`. (Diff: 2/5)
- [ ] `find_closest_k(point, k, max_dist=inf, norm=L2)`: Find the K nearest neighbors within `max_dist`. (Diff: 3/5)

## Efficiency & Rebalancing
- [ ] **Lazy Deletion:** Use the `Node` padding to store a `deleted` flag. Skip these nodes during search and trigger rebalances when tombstone density is high. (Diff: 3/5)
- [ ] **Scapegoat-style Rebalancing:** Instead of global rebalancing, only rebalance specific subtrees that exceed a local balance factor (reduces rebalance frequency). (Diff: 4/5)
- [ ] **Bulk Loading:** Optimized `insert` for large batches of entries. (Diff: 2/5)

## Architecture & Features
- [ ] **Set Mode:** Support `KDTree<PointType, void>` (or tag type) to store only points with no associated values (saves memory). (Diff: 3/5)
- [ ] **3-Dimensional Support:** Generalize `Point` and `KDTree` to support 3D points. (Diff: 2/5)
- [ ] **N-Dimensional Support:** Generalize `Point` and `KDTree` to support arbitrary dimensions beyond 2D. (Diff: 4/5)
- [ ] **Python Binding Coverage:** Expose new search methods and lazy deletion metrics to Python. (Diff: 2/5)

## Node Padding Usage Ideas
On 64-bit systems, the `Node` struct has roughly 4 bytes of padding. Potential uses:
- **Bitflags:** Store `is_deleted`, `axis` (to save a modulo operation), or `is_leaf`.
- **Subtree Count:** Store the number of nodes in the subtree (required for Scapegoat balancing).
- **Check-bit:** Minimal data integrity verification.
