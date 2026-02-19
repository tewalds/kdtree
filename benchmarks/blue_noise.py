# https://blog.demofox.org/2017/10/20/generating-blue-noise-sample-points-with-mitchells-best-candidate-algorithm/

import sys
import time
import json
import random
import kdtree

def run_bench(n):
    bounds = kdtree.Pointd(1000, 1000)
    metric = kdtree.ToroidalL2(kdtree.L2(), bounds)
    tree = kdtree.KDTreed()

    def rand_p():
        return (random.uniform(0, bounds.x), random.uniform(0, bounds.y))

    start = time.perf_counter()
    tree.insert(rand_p(), 0)

    total_query_ms = 0
    query_count = 0

    for i in range(1, n):
        num_candidates = i + 1
        best_p = None
        max_sq_dist = -1

        for _ in range(num_candidates):
            p = rand_p()
            q_start = time.perf_counter()
            closest = tree.find_closest(p, metric=metric)
            total_query_ms += (time.perf_counter() - q_start) * 1000
            query_count += 1

            dx = abs(p[0] - closest.p.x)
            dy = abs(p[1] - closest.p.y)
            dx = min(dx, bounds.x - dx)
            dy = min(dy, bounds.y - dy)
            sq_dist = dx*dx + dy*dy

            if sq_dist > max_sq_dist:
                max_sq_dist = sq_dist
                best_p = p
        tree.insert(best_p, i)

    total_ms = (time.perf_counter() - start) * 1000

    return [
        {"test": "Blue Noise Total", "implementation": "Python", "n": n, "time_ms": total_ms, "iters": 1},
        {"test": "Blue Noise Search", "implementation": "Python", "n": n, "time_ms": total_query_ms, "iters": query_count}
    ]

if __name__ == "__main__":
    sizes = [500, 1000, 2000]
    all_results = []
    for n in sizes:
        all_results.extend(run_bench(n))
    print(json.dumps(all_results, indent=2))
