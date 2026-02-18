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

    assert tree.insert(10, (1, 2))
    assert tree.insert(20, 3, 4)
    assert not tree.empty()
    assert len(tree) == 2

    # Duplicate
    assert not tree.insert(30, (1, 2))
    assert len(tree) == 2

def test_find():
    """Test find operations."""
    tree = kdtree.KDTreed()
    tree.insert(100, (1.0, 2.0))
    tree.insert(200, (3.0, 4.0))

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
    tree.insert(1, (0.0, 0.0))
    tree.insert(2, (1.0, 1.0))
    tree.insert(3, (5.0, 5.0))

    closest = tree.find_closest((1.1, 1.1))
    assert closest.value == 2

    closest = tree.find_closest(0.1, 0.1)
    assert closest.value == 1

def test_norm_parameter():
    """Test L1 vs L2 distance."""
    tree = kdtree.KDTreed()
    tree.insert(1, (0.0, 0.0))
    tree.insert(2, (1.0, 0.0))
    tree.insert(3, (0.0, 1.0))

    # Both work
    result_l2 = tree.find_closest((0.5, 0.5), kdtree.Norm.L2)
    result_l1 = tree.find_closest((0.5, 0.5), kdtree.Norm.L1)

    assert result_l2.value in [1, 2, 3]
    assert result_l1.value in [1, 2, 3]

def test_remove():
    """Test removal."""
    tree = kdtree.KDTreei()
    tree.insert(10, (1, 2))
    tree.insert(20, (3, 4))

    assert len(tree) == 2
    assert tree.remove((1, 2))
    assert len(tree) == 1
    assert not tree.exists(1, 2)

    assert not tree.remove((1, 2))
    assert len(tree) == 1

def test_pop_closest():
    """Test pop_closest."""
    tree = kdtree.KDTreei()
    tree.insert(1, (0, 0))
    tree.insert(2, (1, 1))
    tree.insert(3, (5, 5))

    closest = tree.pop_closest((1, 1))
    assert closest.value == 2
    assert len(tree) == 2
    assert not tree.exists((1, 1))

def test_iteration():
    """Test iteration."""
    tree = kdtree.KDTreei()
    tree.insert(1, (1, 2))
    tree.insert(2, (3, 4))
    tree.insert(3, (5, 6))

    values = list(tree)
    assert len(values) == 3

    value_ids = {v.value for v in values}
    assert value_ids == {1, 2, 3}

def test_clear():
    """Test clear."""
    tree = kdtree.KDTreei()
    tree.insert(1, (1, 2))
    tree.insert(2, (3, 4))

    assert len(tree) == 2
    tree.clear()
    assert tree.empty()

def test_rebalance():
    """Test rebalancing."""
    tree = kdtree.KDTreei()
    for i in range(100):
        tree.insert(i, (i, i*2))

    tree.rebalance()
    assert len(tree) == 100

def test_python_object_storage():
    """Test storing Python objects."""
    tree = kdtree.KDTreePyd()
    tree.insert({"name": "Alice", "score": 100}, (1.0, 2.0))
    tree.insert({"name": "Bob", "score": 85}, (3.0, 4.0))

    result = tree.find_closest((2.0, 3.0))
    assert result.value["name"] in ["Alice", "Bob"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
