import sys
import time
import json
import numpy as np
from scipy.spatial import KDTree

def run_bench(n):
    np.random.seed(42)
    data = np.random.uniform(0, 1000, size=(n, 2))
    
    start_build = time.perf_counter()
    tree = KDTree(data)
    end_build = time.perf_counter()
    build_ms = (end_build - start_build) * 1000
    
    num_queries = 1000
    queries = np.random.uniform(0, 1000, size=(num_queries, 2))
    
    start_query = time.perf_counter()
    for q in queries:
        tree.query(q)
    end_query = time.perf_counter()
    query_ms = (end_query - start_query) * 1000
    
    return [
        {"test": "Core Build", "implementation": "SciPy", "n": n, "time_ms": build_ms, "iters": 1},
        {"test": "Core Query (1k)", "implementation": "SciPy", "n": n, "time_ms": query_ms, "iters": num_queries}
    ]

if __name__ == "__main__":
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    all_results = []
    for n in sizes:
        all_results.extend(run_bench(n))
    print(json.dumps(all_results, indent=2))
