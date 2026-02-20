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
    assert tree.insert(kdtree.Pointi(3, 4), 20)
    assert not tree.insert((1, 2), 30)  # Duplicate

    assert len(tree) == 2
    assert not tree.empty()

def test_find():
    """Test find operations."""
    tree = kdtree.KDTreed()
    tree.insert((1.0, 2.0), 100)
    tree.insert(kdtree.Pointd(3.0, 4.0), 200)

    result = tree.find((1.0, 2.0))
    assert result is not None
    assert result.value == 100
    assert result.p.x == 1.0
    assert result.p.y == 2.0

    assert tree.find((5.0, 6.0)) is None
    assert tree.exists((3.0, 4.0))
    assert not tree.exists((0.0, 0.0))

def test_find_closest():
    """Test nearest neighbor search."""
    tree = kdtree.KDTreed()
    tree.insert((0.0, 0.0), 1)
    tree.insert((1.0, 1.0), 2)
    tree.insert((2.0, 2.0), 3)

    closest = tree.find_closest((0.1, 0.1))
    assert closest.value == 1

    closest = tree.find_closest((1.1, 1.1))
    assert closest.value == 2

    # Verify we can still pass Point objects
    p = kdtree.Pointd(0.1, 0.1)
    closest = tree.find_closest(p)
    assert closest.value == 1

def test_find_closest_with_max_dist():
    """Test nearest neighbor with distance limit."""
    tree = kdtree.KDTreed()
    tree.insert((0.0, 0.0), 1)
    tree.insert((10.0, 10.0), 2)

    # Note: Signature is find_closest(point, metric=None, max_dist=None)
    # L1 (Manhattan): dist((1, 1), (0, 0)) = 1+1 = 2
    assert tree.find_closest((1, 1), kdtree.L1(), 3) is not None
    assert tree.find_closest((1, 1), kdtree.L1(), 2) is not None
    assert tree.find_closest((1, 1), kdtree.L1(), 1) is None

    # L2sq (Squared Euclidian): dist((3, 4), (0, 0)) = 25
    assert tree.find_closest((3, 4), kdtree.L2sq(), 26) is not None
    assert tree.find_closest((3, 4), kdtree.L2sq(), 25) is not None
    assert tree.find_closest((3, 4), kdtree.L2sq(), 24) is None

    # Linf (Chebyshev): dist((2, 2), (0, 0)) = max(2, 2) = 2
    assert tree.find_closest((2, 2), kdtree.Linf(), 3) is not None
    assert tree.find_closest((2, 2), kdtree.Linf(), 2) is not None
    assert tree.find_closest((2, 2), kdtree.Linf(), 1) is None

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
    # Note: Signature is find_closest_k(point, k=1, metric=None, max_dist=None)
    res = tree.find_closest_k((0.0, 0.0), 10, kdtree.L1(), 4.0)
    assert len(res) == 3
    ids = {e.value for e in res}
    assert ids == {1, 2, 3}

    # No results within limit
    res = tree.find_closest_k((0.5, 0.5), 10, kdtree.L2sq(), 0.1)
    assert len(res) == 0

def test_norm_parameter():
    """Test L1 vs L2 distance."""
    tree = kdtree.KDTreed()
    # Insert points where L1 and L2 nearest differ
    tree.insert((10, 0), 1)
    tree.insert((9, 4), 2)
    tree.insert((7, 7), 3)

    # Point (0,0):
    # L1: (10,0)=10, (9,4)=13, (7,7)=14 -> 1 is closest
    # L2: (10,0)=10, (9,4)=9.85, (7,7)=9.9 -> 2 is closest
    # Linf: (10,0)=10, (9,4)=9, (7,7)=7 -> 3 is closest

    assert tree.find_closest((0, 0), kdtree.L1()).value == 1
    assert tree.find_closest((0, 0), kdtree.L2()).value == 2
    assert tree.find_closest((0, 0), kdtree.L2sq()).value == 2
    assert tree.find_closest((0, 0), kdtree.Linf()).value == 3



def test_remove():
    """Test removal."""
    tree = kdtree.KDTreei()
    tree.insert((1, 2), 10)
    tree.insert((3, 4), 20)

    assert len(tree) == 2
    assert tree.remove((1, 2))
    assert len(tree) == 1
    assert not tree.exists((1, 2))
    assert tree.exists((3, 4))

    assert not tree.remove((5, 6))  # Non-existent
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
def test_clear():
    """Test clear."""
    tree = kdtree.KDTreei()
    tree.insert((1, 2), 1)
    tree.insert((3, 4), 2)

    assert len(tree) == 2
    tree.clear()
    assert tree.empty()
    assert len(tree) == 0


def test_rebalance():
    """Test rebalancing."""
    tree = kdtree.KDTreei()
    for i in range(100):
        tree.insert((i, i*2), i)

    tree.rebalance()
    assert len(tree) == 100

def test_iterator():
    """Test iteration."""
    tree = kdtree.KDTreei()
    data = [((1, 1), 1), ((2, 2), 2), ((3, 3), 3)]
    for p, v in data:
        tree.insert(p, v)

    results = []
    for entry in tree:
        results.append(((entry.p.x, entry.p.y), entry.value))

    assert len(results) == 3
    # Order might vary depending on tree structure
    for entry in data:
        assert entry in results

def test_buffer_interface():
    """Test optimized buffer construction."""
    import numpy as np
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    values = np.array([10, 20, 30], dtype=np.int64)

    tree = kdtree.KDTreed(coords, values)
    assert len(tree) == 3
    assert tree.find((1.0, 1.0)).value == 20

def test_python_objects():
    """Test storing arbitrary Python objects."""
    tree = kdtree.KDTreePyd()
    tree.insert((1.0, 2.0), {"name": "Alice"})
    tree.insert((3.0, 4.0), ["Bob", 42])

    assert tree.find((1.0, 2.0)).value["name"] == "Alice"
    assert tree.find((3.0, 4.0)).value[0] == "Bob"

def test_toroidal():
    """Test toroidal (wraparound) distance."""
    bounds = kdtree.Pointd(100, 100)
    metric = kdtree.ToroidalL2(bounds)
    tree = kdtree.KDTreed()

    # Point at (1,1) and (99,99)
    tree.insert((1, 1), 1)
    tree.insert((99, 99), 2)

    # Standard Euclidean (1,1) is closer to (0,0) than (99,99)
    # But Toroidal (99,99) is only dist 1 from boundary
    res = tree.find_closest((0, 0), metric, 10.0)
    assert res.value == 2 or res.value == 1 # Both are dist sqrt(2)

    # Specifically check wraparound
    tree.clear()
    tree.insert((1, 1), 1)
    tree.insert((10, 10), 2)
    res = tree.find_closest((99, 99), metric, 10.0)
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
    found_l2 = tree.find_all_within(query, kdtree.L2(), radius)
    expected_l2 = [p for p, i in points if (p[0]**2 + p[1]**2) <= radius**2]
    assert len(found_l2) == len(expected_l2)
    for entry in found_l2:
        assert (entry.p.x**2 + entry.p.y**2) <= radius**2 + 1e-7

    # 2. L1
    found_l1 = tree.find_all_within(query, kdtree.L1(), radius)
    expected_l1 = [p for p, i in points if (abs(p[0]) + abs(p[1])) <= radius]
    assert len(found_l1) == len(expected_l1)

    # 3. Linf
    found_linf = tree.find_all_within(query, kdtree.Linf(), radius)
    expected_linf = [p for p, i in points if max(abs(p[0]), abs(p[1])) <= radius]
    assert len(found_linf) == len(expected_linf)

def test_metric_enforcement():
    """Test that metric is required when max_dist or radius is provided."""
    tree = kdtree.KDTreed()
    tree.insert((0, 0), 1)

    # find_closest with max_dist but no metric
    with pytest.raises(ValueError, match="Metric must be specified"):
        tree.find_closest((0, 0), max_dist=10.0)

    # find_closest_k with max_dist but no metric
    with pytest.raises(ValueError, match="Metric must be specified"):
        tree.find_closest_k((0, 0), 10, max_dist=10.0)

    # find_all_within without metric
    with pytest.raises(TypeError):
        tree.find_all_within((0, 0), radius=10.0)

def test_metric_dist():
    """Test the dist() method on metric objects."""
    p1 = (0, 0)
    p2 = (3, 4)

    assert kdtree.L1().dist(p1, p2) == 7.0
    assert kdtree.L2().dist(p1, p2) == 5.0
    assert kdtree.L2sq().dist(p1, p2) == 25.0
    assert kdtree.Linf().dist(p1, p2) == 4.0

    # Toroidal
    bounds = kdtree.Pointd(10, 10)
    # (0,0) to (9,9) is (1,1) across boundary
    assert kdtree.ToroidalL2(bounds).dist((0, 0), (9, 9)) == pytest.approx(2**0.5)
    assert kdtree.ToroidalL2sq(bounds).dist((0, 0), (9, 9)) == pytest.approx(2.0)

    # Great Circle
    # SF to LA is ~559km
    sf = (37.7749, -122.4194)
    la = (34.0522, -118.2437)
    d = kdtree.GreatCircle().dist(sf, la)
    assert 550000 < d < 570000

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
    all_res = tree.find_all_within(sf, metric, 600000)
    assert len(all_res) == 2

    # 10km radius from SF should find only SF
    all_res = tree.find_all_within(sf, metric, 10000)
    assert len(all_res) == 1
    assert all_res[0].value == "SF"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
