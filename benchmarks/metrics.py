import sys
import time
import json
import numpy as np
import kdtree

def run_metric_bench(tree, queries, metric_name, metric, n):
    results = []
    num_queries = len(queries)

    # 1. find_closest
    start = time.perf_counter()
    for q in queries:
        tree.find_closest(q, metric=metric)
    ms = (time.perf_counter() - start) * 1000
    results.append({"test": f"Query: find_closest ({metric_name})", "implementation": "Python", "n": n, "time_ms": ms, "iters": num_queries})

    # 2. find_closest_k (k=5)
    start = time.perf_counter()
    for q in queries:
        tree.find_closest_k(q, 5, metric=metric)
    ms = (time.perf_counter() - start) * 1000
    results.append({"test": f"Query: find_closest_k=5 ({metric_name})", "implementation": "Python", "n": n, "time_ms": ms, "iters": num_queries})

    # 3. find_all_within
    radius = 10.0
    if metric_name == "GreatCircle": radius = 100000.0 # 100km
    start = time.perf_counter()
    for q in queries:
        tree.find_all_within(q, radius, metric=metric)
    ms = (time.perf_counter() - start) * 1000
    results.append({"test": f"Query: find_all_within ({metric_name})", "implementation": "Python", "n": n, "time_ms": ms, "iters": num_queries})

    return results

def main():
    sizes = [1000, 10000, 100000, 1000000]
    all_res = []

    for n in sizes:
        np.random.seed(42)
        data = np.random.uniform(0, 1000, size=(n, 2))
        tree = kdtree.KDTreed(data, np.arange(n, dtype=np.int64))

        queries = np.random.uniform(0, 1000, size=(1000, 2)).tolist()

        all_res.extend(run_metric_bench(tree, queries, "L1", kdtree.L1(), n))
        all_res.extend(run_metric_bench(tree, queries, "L2", kdtree.L2(), n))
        all_res.extend(run_metric_bench(tree, queries, "Linf", kdtree.Linf(), n))
        all_res.extend(run_metric_bench(tree, queries, "ToroidalL2", kdtree.ToroidalL2(kdtree.L2(), kdtree.Pointd(1000, 1000)), n))
        all_res.extend(run_metric_bench(tree, queries, "GreatCircle", kdtree.GreatCircle(), n))

    print(json.dumps(all_res, indent=2))

if __name__ == "__main__":
    main()
