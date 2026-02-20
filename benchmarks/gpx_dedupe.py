import sys
import time
import json
import numpy as np
import kdtree
import os
from scipy.spatial import KDTree as ScipyKDTree

def run_all():
    tsv_file = "benchmarks/cdt.tsv"
    if not os.path.exists(tsv_file):
        return []

    points = np.loadtxt(tsv_file)
    n = len(points)
    results = []
    
    # 1. Bulk
    start = time.perf_counter()
    tree = kdtree.KDTreed(points, np.arange(n, dtype=np.int64))
    results.append({"test": "GPX Bulk Build", "implementation": "Python", "n": n, "time_ms": (time.perf_counter()-start)*1000})
    
    start = time.perf_counter()
    sp_tree = ScipyKDTree(points)
    results.append({"test": "GPX Bulk Build", "implementation": "SciPy", "n": n, "time_ms": (time.perf_counter()-start)*1000})

    # 2. Incremental
    start = time.perf_counter()
    tree_inc = kdtree.KDTreed()
    for i in range(n):
        tree_inc.insert(points[i], i)
    results.append({"test": "GPX Inc Load", "implementation": "Python", "n": n, "time_ms": (time.perf_counter()-start)*1000})

    # 3. Dedupe
    thresholds = [("1m", 0.00001), ("10m", 0.0001), ("100m", 0.001)]
    for label, val in thresholds:
        start = time.perf_counter()
        tree_dedupe = kdtree.KDTreed()
        kept = 0
        for p in points:
            if not tree_dedupe.find_closest(p, kdtree.L2sq(), val * val):
                tree_dedupe.insert(p, kept)
                kept += 1
        results.append({"test": f"GPX Dedupe ({label})", "implementation": "Python", "n": n, "time_ms": (time.perf_counter()-start)*1000})

    return results

if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
