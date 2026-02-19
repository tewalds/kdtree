#!/usr/bin/env python3
"""Tests for KDTree Python bindings."""

import pytest
import kdtree

def test_point_creation():
    """Test Point creation."""
    p1 = kdtree.Pointi(1, 2)
    assert p1.x == 1 and p1.y == 2

    p2 = kdtree.Pointd(1.5, 2.5)
    assert p2.x == 1.5 and p2.y == 2.5

    p3 = kdtree.Pointi((3, 4))
    assert p3.x == 3 and p3.y == 4

def test_tree_basic():
    """Test basic operations."""
    tree = kdtree.KDTreei()
    assert tree.empty()
    assert len(tree) == 0

    assert tree.insert((1, 2), 10)
    assert tree.insert(3, 4, 20)
    assert not tree.empty()
    assert len(tree) == 2

    # Duplicate
    assert not tree.insert((1, 2), 30)
    assert len(tree) == 2

def test_find():
    """Test find operations."""
    tree = kdtree.KDTreed()
    tree.insert((1.0, 2.0), 100)
    tree.insert((3.0, 4.0), 200)

    result = tree.find(1.0, 2.0)
    assert result is not None
    assert result.value == 100

    result = tree.find((10.0, 10.0))
    assert result is None

    assert tree.exists((1.0, 2.0))
    assert not tree.exists(10.0, 10.0)

def test_find_closest():
    """Test nearest neighbor."""
    tree = kdtree.KDTreed()
    tree.insert((0.0, 0.0), 1)
    tree.insert((1.0, 1.0), 2)
    tree.insert((5.0, 5.0), 3)

    closest = tree.find_closest((1.1, 1.1))
    assert closest.value == 2

    closest = tree.find_closest(0.1, 0.1)
    assert closest.value == 1

def test_find_closest_with_max_dist():
    """Test nearest neighbor with distance limit."""
    tree = kdtree.KDTreed()
    tree.insert((0.0, 0.0), 1)
    tree.insert((10.0, 10.0), 2)

    # L1 (Manhattan): dist((1, 1), (0, 0)) = 1+1 = 2
    assert tree.find_closest((1, 1), 3, kdtree.L1()) is not None
    assert tree.find_closest((1, 1), 2, kdtree.L1()) is not None
    assert tree.find_closest((1, 1), 1, kdtree.L1()) is None

    # L2 (Euclidian): dist((3, 4), (0, 0)) = 5
    assert tree.find_closest((3, 4), 6, kdtree.L2()) is not None
    assert tree.find_closest((3, 4), 5, kdtree.L2()) is not None
    assert tree.find_closest((3, 4), 4, kdtree.L2()) is None

    # Linf (Chebyshev): dist((2, 2), (0, 0)) = max(2, 2) = 2
    assert tree.find_closest((2, 2), 3, kdtree.Linf()) is not None
    assert tree.find_closest((2, 2), 2, kdtree.Linf()) is not None
    assert tree.find_closest((2, 2), 1, kdtree.Linf()) is None

def test_find_closest_k():
    """Test finding K nearest neighbors."""
    tree = kdtree.KDTreed()
    tree.insert((0.0, 0.0), 1)
    tree.insert((1.0, 1.0), 2)
    tree.insert((2.0, 2.0), 3)
    tree.insert((10.0, 10.0), 4)

    # Basic K=2
    res = tree.find_closest_k((0.5, 0.5), 2)
    assert len(res) == 2
    ids = {e.value for e in res}
    assert ids == {1, 2}

    # K=10 with limit
    # dist((0,0), (2,2)) is 4.0 in L1
    res = tree.find_closest_k((0.0, 0.0), 10, 4.0, kdtree.L1())
    assert len(res) == 3
    ids = {e.value for e in res}
    assert ids == {1, 2, 3}

    # No results within limit
    res = tree.find_closest_k((0.5, 0.5), 10, 0.1)
    assert len(res) == 0

def test_norm_parameter():
    """Test L1 vs L2 distance."""
    tree = kdtree.KDTreed()
    tree.insert((0.0, 0.0), 1)
    tree.insert((1.0, 0.0), 2)
    tree.insert((0.0, 1.0), 3)

    # Both work
    result_l2 = tree.find_closest((0.5, 0.5), kdtree.L2())
    result_l1 = tree.find_closest((0.5, 0.5), kdtree.L1())

    assert result_l2.value in [1, 2, 3]
    assert result_l1.value in [1, 2, 3]

def test_remove():
    """Test removal."""
    tree = kdtree.KDTreei()
    tree.insert((1, 2), 10)
    tree.insert((3, 4), 20)

    assert len(tree) == 2
    assert tree.remove((1, 2))
    assert len(tree) == 1
    assert not tree.exists(1, 2)

    assert not tree.remove((1, 2))
    assert len(tree) == 1

def test_pop_closest():
    """Test pop_closest."""
    tree = kdtree.KDTreei()
    tree.insert((0, 0), 1)
    tree.insert((1, 1), 2)
    tree.insert((5, 5), 3)

    closest = tree.pop_closest((1, 1))
    assert closest.value == 2
    assert len(tree) == 2
    assert not tree.exists((1, 1))

def test_iteration():
    """Test iteration."""
    tree = kdtree.KDTreei()
    tree.insert((1, 2), 1)
    tree.insert((3, 4), 2)
    tree.insert((5, 6), 3)

    entries = list(tree)
    assert len(entries) == 3

    entry_ids = {e.value for e in entries}
    assert entry_ids == {1, 2, 3}

def test_clear():
    """Test clear."""
    tree = kdtree.KDTreei()
    tree.insert((1, 2), 1)
    tree.insert((3, 4), 2)

    assert len(tree) == 2
    tree.clear()
    assert tree.empty()

def test_rebalance():
    """Test rebalancing."""
    tree = kdtree.KDTreei()
    for i in range(100):
        tree.insert((i, i*2), i)

    tree.rebalance()
    assert len(tree) == 100

def test_python_object_storage():
    """Test storing Python objects."""
    tree = kdtree.KDTreePyd()
    tree.insert((1.0, 2.0), {"name": "Alice", "score": 100})
    tree.insert((3.0, 4.0), {"name": "Bob", "score": 85})

    result = tree.find_closest((2.0, 3.0))
    assert result.value["name"] in ["Alice", "Bob"]

def test_toroidal():
    """Test toroidal distance."""
    # Domain 100x100
    bounds = kdtree.Pointd(100, 100)
    metric = kdtree.ToroidalL2(kdtree.L2(), bounds)
    tree = kdtree.KDTreed()

    # Point at (1,1) and (99,99)
    tree.insert((1, 1), 1)
    tree.insert((99, 99), 2)

    # Standard Euclidean (1,1) is closer to (0,0) than (99,99)
    # But Toroidal (99,99) is only dist 1 from boundary
    res = tree.find_closest((0, 0), 10.0, metric)
    assert res.value == 2 or res.value == 1 # Both are dist sqrt(2)

    # Specifically check wraparound
    tree.clear()
    tree.insert((1, 1), 1)
    tree.insert((10, 10), 2)
    res = tree.find_closest((99, 99), 10.0, metric)
    assert res.value == 1 # (99,99) to (1,1) is dist 2 in toroidal

def test_find_all_within():
    """Test finding all points within a radius."""
    import random
    random.seed(42)
    tree = kdtree.KDTreed()
    points = []
    for i in range(1000):
        p = (random.uniform(-100, 100), random.uniform(-100, 100))
        if tree.insert(p, i):
            points.append((p, i))

    query = (0, 0)
    radius = 50.0

    # 1. L2
    found_l2 = tree.find_all_within(query, radius, kdtree.L2())
    expected_l2 = [p for p, i in points if (p[0]**2 + p[1]**2) <= radius**2]
    assert len(found_l2) == len(expected_l2)
    for entry in found_l2:
        assert (entry.p.x**2 + entry.p.y**2) <= radius**2 + 1e-7

    # 2. L1
    found_l1 = tree.find_all_within(query, radius, kdtree.L1())
    expected_l1 = [p for p, i in points if (abs(p[0]) + abs(p[1])) <= radius]
    assert len(found_l1) == len(expected_l1)

    # 3. Linf
    found_linf = tree.find_all_within(query, radius, kdtree.Linf())
    expected_linf = [p for p, i in points if max(abs(p[0]), abs(p[1])) <= radius]
    assert len(found_linf) == len(expected_linf)

def test_great_circle():
    """Test Great Circle distance."""
    # San Francisco and Los Angeles
    sf = (37.7749, -122.4194)
    la = (34.0522, -118.2437)

    tree = kdtree.KDTreePyd()
    tree.insert(sf, "SF")
    tree.insert(la, "LA")

    metric = kdtree.GreatCircle()

    # Should find SF
    res = tree.find_closest(sf, metric=metric)
    assert res.value == "SF"

    # Distance is roughly 550km. 600km radius should find both.
    all_res = tree.find_all_within(sf, 600000, metric=metric)
    assert len(all_res) == 2

    # 10km radius from SF should find only SF
    all_res = tree.find_all_within(sf, 10000, metric=metric)
    assert len(all_res) == 1
    assert all_res[0].value == "SF"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
