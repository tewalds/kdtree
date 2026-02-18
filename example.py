#!/usr/bin/env python3
"""Example usage of the KDTree Python bindings."""

import kdtree


def basic():
    print("=== Basic Usage with int64_t Storage ===\n")

    # Create tree with double coordinates, int64 values
    tree = kdtree.KDTreed()

    # Insert using different convenience methods
    tree.insert(1, (1.5, 2.3))           # tuple
    tree.insert(2, [4.1, 3.7])           # list
    tree.insert(3, kdtree.Pointd(0.8, 5.2))  # explicit Point
    tree.insert(4, 3.0, 4.0)             # separate coords

    print(f"Tree size: {len(tree)}")
    print(f"Stats: {tree.balance_str()}\n")

    # Query using different methods
    print("Exists checks:")
    print(f"  (1.5, 2.3): {tree.exists((1.5, 2.3))}")
    print(f"  (1.5, 2.3): {tree.exists(1.5, 2.3)}")
    print(f"  (100, 100): {tree.exists(100, 100)}\n")

    # Find exact
    result = tree.find(1.5, 2.3)
    if result:
        print(f"Found: {result}\n")

    # Find closest with different norms
    query = (3.5, 4.5)
    closest_l2 = tree.find_closest(query)  # default L2
    closest_l1 = tree.find_closest(query, kdtree.Norm.L1)
    closest_linf = tree.find_closest(query, kdtree.Norm.Linf)

    print(f"Closest to {query}:")
    print(f"  L2 (Euclidean): {closest_l2}")
    print(f"  L1 (Manhattan): {closest_l1}\n")
    print(f"  Linf (King):    {closest_linf}\n")

    # Iterate
    print("All values:")
    for value in tree:
        print(f"  {value}")

    print("\n=== Python Object Storage ===\n")

    # Store arbitrary Python objects
    pytree = kdtree.KDTreePyd()

    pytree.insert({"name": "Alice", "score": 100}, (1.0, 2.0))
    pytree.insert({"name": "Bob", "score": 85}, (3.0, 4.0))
    pytree.insert(["some", "list"], (5.0, 1.0))

    # Find closest
    result = pytree.find_closest((2.0, 3.0))
    print(f"Closest object to (2.0, 3.0): {result.value}")
    print(f"  at position: {result.p}\n")

    # Remove and iterate
    pytree.remove(3.0, 4.0)
    print(f"After removing (3.0, 4.0), remaining objects:")
    for val in pytree:
        print(f"  {val.value} at {val.p}")

    print("\n=== Manhattan Distance Example ===\n")

    # Use integer coordinates with L1 distance
    grid = kdtree.KDTreei()

    # Create a grid of points
    for x in range(5):
        for y in range(5):
            grid.insert(x * 5 + y, x, y)

    center = (2, 2)
    print(f"Points nearest to {center} (Manhattan distance):")
    for _ in range(5):
        if grid.empty():
            break
        nearest = grid.pop_closest(center, kdtree.Norm.L1)
        dist = abs(nearest.p.x - center[0]) + abs(nearest.p.y - center[1])
        print(f"  {nearest.p} (distance: {dist})")


def minesweeper_spatial_queue():
    """
    Minesweeper: Spatial task queue that always opens the cell closest to the last one.
    This creates a more natural "flood fill" pattern when revealing cells.
    """
    print("=== Minesweeper Spatial Queue ===\n")

    # Track cells to reveal with their priorities/distances
    to_reveal = kdtree.KDTreei()

    # Starting point: player clicks (5, 5)
    last_revealed = (5, 5)
    print(f"Initial click: {last_revealed}")

    # Reveal creates new cells to check (neighbors)
    # Priority based on distance from last revealed
    neighbors = [(4,5), (6,5), (5,4), (5,6), (4,4), (6,6), (4,6), (6,4)]
    for x, y in neighbors:
        # Insert with distance as "priority" - we'll use position for nearest query
        to_reveal.insert(0, x, y)  # 0 = priority, will be overridden by spatial distance

    print(f"Added {len(to_reveal)} neighbors to queue")

    # Process queue: always take the cell closest to the last revealed
    revealed_order = [last_revealed]
    iterations = 0
    max_iterations = 8

    while not to_reveal.empty() and iterations < max_iterations:
        # Get cell closest to last revealed position
        next_cell = to_reveal.pop_closest(last_revealed)
        last_revealed = (next_cell.p.x, next_cell.p.y)
        revealed_order.append(last_revealed)
        iterations += 1

        # In real minesweeper, we'd add more neighbors here if it's a 0
        # For demo, just show the order

    print(f"Reveal order: {' -> '.join(str(p) for p in revealed_order)}")
    print(f"Natural flood-fill pattern from {revealed_order[0]} outward\n")


def gps_track_deduplication():
    """
    GPS Track Deduplication: Add points one at a time, reusing nearby points.
    Useful for simplifying GPS tracks without creating duplicates at the same location.
    """
    print("=== GPS Track Deduplication ===\n")

    # Store unique GPS points (using double precision)
    unique_points = kdtree.KDTreePyd()

    # Simulated GPS track with some noise and duplicates
    raw_track = [
        (37.7749, -122.4194),  # San Francisco
        (37.7750, -122.4195),  # ~11m away (noise)
        (37.7751, -122.4194),  # ~11m away (noise)
        (37.7849, -122.4094),  # ~1.5km away
        (37.7850, -122.4095),  # ~11m away (noise)
        (37.7949, -122.3994),  # ~1.5km away
        (37.7950, -122.3995),  # ~11m away (noise)
        (37.7949, -122.3994),  # Exact duplicate
    ]

    # Deduplicate: only add if no point within threshold
    threshold_km = 0.1  # 100 meters
    # Rough conversion: 1 degree ≈ 111km at equator
    threshold_degrees = threshold_km / 111.0

    deduplicated_track = []

    for i, (lat, lon) in enumerate(raw_track):
        if unique_points.empty():
            # First point always added
            point_id = len(deduplicated_track)
            unique_points.insert({"id": point_id, "lat": lat, "lon": lon}, (lat, lon))
            deduplicated_track.append((lat, lon))
            print(f"Point {i}: ({lat:.4f}, {lon:.4f}) - Added (first point)")
        else:
            # Find closest existing point
            closest = unique_points.find_closest((lat, lon))

            # Calculate approximate distance (Euclidean in degrees)
            dist_deg = ((closest.p.x - lat)**2 + (closest.p.y - lon)**2)**0.5
            dist_km = dist_deg * 111.0

            if dist_deg < threshold_degrees:
                # Too close to existing point, reuse it
                reused_point = (closest.p.x, closest.p.y)
                print(f"Point {i}: ({lat:.4f}, {lon:.4f}) - Reused {reused_point} ({dist_km:.1f}m away)")
            else:
                # Far enough, add new point
                point_id = len(deduplicated_track)
                unique_points.insert({"id": point_id, "lat": lat, "lon": lon}, (lat, lon))
                deduplicated_track.append((lat, lon))
                print(f"Point {i}: ({lat:.4f}, {lon:.4f}) - Added ({dist_km:.1f}km from nearest)")

    print(f"\nTrack simplified: {len(raw_track)} points → {len(deduplicated_track)} unique points")
    print(f"Reduction: {(1 - len(deduplicated_track)/len(raw_track))*100:.0f}%\n")


def spatial_proximity_search():
    """
    Quick demo: Using both L1 and L2 distance on the same tree.
    """
    print("=== Spatial Proximity: L1 vs L2 Distance ===\n")

    # City locations (approximate)
    cities = kdtree.KDTreed()
    cities.insert(1, (37.77, -122.42))  # San Francisco
    cities.insert(2, (34.05, -118.24))  # Los Angeles
    cities.insert(3, (37.34, -121.89))  # San Jose
    cities.insert(4, (38.58, -121.49))  # Sacramento

    query = (37.50, -122.00)  # Somewhere in the Bay Area

    print(f"Query location: {query}")

    # L2 (Euclidean): Geometric distance
    closest_l2 = cities.find_closest(query, kdtree.Norm.L2)
    print(f"Closest (L2/Euclidean): City {closest_l2.value} at {closest_l2.p}")

    # L1 (Manhattan): Grid/taxicab distance
    closest_l1 = cities.find_closest(query, kdtree.Norm.L1)
    print(f"Closest (L1/Manhattan): City {closest_l1.value} at {closest_l1.p}")

    print("\nNote: Same tree, different distance metrics!")
    print()

def main():
    basic()
    minesweeper_spatial_queue()
    gps_track_deduplication()
    spatial_proximity_search()

if __name__ == "__main__":
    main()
