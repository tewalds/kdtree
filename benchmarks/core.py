import sys
import time
import json
import numpy as np
import kdtree

def run_bench(n):
    np.random.seed(42)
    data_coords = np.random.uniform(0, 1000, size=(n, 2))
    data_values = np.arange(n, dtype=np.int64)
    
    results = []

    # 1. KDTreed (Double coords, Int64 values)
    start_build = time.perf_counter()
    tree = kdtree.KDTreed(data_coords, data_values)
    end_build = time.perf_counter()
    
    num_queries = 1000
    queries = np.random.uniform(0, 1000, size=(num_queries, 2))
    
    start_query = time.perf_counter()
    for q in queries:
        tree.find_closest(q.tolist())
    end_query = time.perf_counter()
    
    results.extend([
        {"test": "Core Build", "implementation": "Python (float/int64)", "n": n, "time_ms": (end_build - start_build) * 1000, "iters": 1},
        {"test": "Core Query (1k)", "implementation": "Python (float/int64)", "n": n, "time_ms": (end_query - start_query) * 1000, "iters": num_queries}
    ])

    # 2. KDTreePyd (Double coords, Python Object values)
    # Bulk loading objects might be different, let's use a list of objects
    objects = [{"id": i} for i in range(n)]
    
    start_build = time.perf_counter()
    # KDTreePyd bulk constructor handles numpy coords and list of objects
    tree_py = kdtree.KDTreePyd(data_coords, objects)
    end_build = time.perf_counter()
    
    start_query = time.perf_counter()
    for q in queries:
        tree_py.find_closest(q.tolist())
    end_query = time.perf_counter()
    
    results.extend([
        {"test": "Core Build", "implementation": "Python (float/object)", "n": n, "time_ms": (end_build - start_build) * 1000, "iters": 1},
        {"test": "Core Query (1k)", "implementation": "Python (float/object)", "n": n, "time_ms": (end_query - start_query) * 1000, "iters": num_queries}
    ])
    
    return results

if __name__ == "__main__":
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    all_results = []
    for n in sizes:
        all_results.extend(run_bench(n))
    print(json.dumps(all_results, indent=2))
