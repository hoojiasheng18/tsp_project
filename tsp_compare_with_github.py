#!/usr/bin/env python3
"""
tsp_compare_from_github.py

Port of GeneticAlgorithm-TSP (parano) JS -> Python plus Nearest-Neighbor and Held-Karp
Usage:
    python tsp_compare_from_github.py [--js-data-file PATH_TO_DATA_JS]

Outputs:
 - tsp_results.csv
 - figures/instance_<name>_comparison.png (requires matplotlib)
"""

import re, json, math, random, time, argparse, os, csv, statistics
from typing import List, Tuple, Sequence, Dict
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

Point = Tuple[float, float]

# -------------------------
# helper: load arrays from a JS file like data.js which contains:
# data200 = [{ "x":..,"y":.. }, ...];
# We'll extract any variable named data\d+ and convert to list of points.
# -------------------------
def load_points_from_js(js_path: str) -> dict:
    with open(js_path, 'r', encoding='utf-8') as f:
        txt = f.read()
    out = {}
    # regex: find data arrays like data40, data200, data500
    pattern = r'(data\d+)\s*=\s*(\[[^\]]*\])'
    matches = re.findall(pattern, txt, flags=re.MULTILINE | re.DOTALL)
    for name, array_text in matches:
        try:
            # Remove trailing commas
            array_text = re.sub(r',\s*([\]}])', r'\1', array_text)
            # Load JSON
            arr = json.loads(array_text)
            pts = [(float(item['x']), float(item['y'])) for item in arr]
            out[name] = pts
        except Exception:
            # fallback: grab all numbers
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', array_text)
            coords = [(float(nums[i]), float(nums[i+1])) for i in range(0, len(nums), 2)]
            out[name] = coords
    return out

# -------------------------
# distance and evaluation (matching evaluate in JS)
# JS evaluate sums dis[ind[0]][ind[last]] + consecutive pairs
# We'll use Euclidean and keep same ordering.
# -------------------------
def euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def build_distance_matrix(points: Sequence[Point]) -> List[List[float]]:
    n = len(points)
    dis = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j:
                dis[i][j]=0.0
            else:
                dis[i][j] = euclid(points[i], points[j])
    return dis

def evaluate_tour_by_index(ind: Sequence[int], dis: List[List[float]]) -> float:
    if not ind:
        return 0.0
    s = dis[ind[0]][ind[-1]]
    for i in range(1, len(ind)):
        s += dis[ind[i]][ind[i-1]]
    return s

# -------------------------
# Held-Karp (exact) - same as earlier scripts
# -------------------------
def held_karp(points: Sequence[Point]) -> Tuple[float, List[int]]:
    n = len(points)
    if n==0:
        return 0.0, []
    dis = build_distance_matrix(points)
    if n==1:
        return 0.0, [0]
    # DP over masks of nodes 1..n-1, fix start=0
    dp = {}
    for k in range(1, n):
        mask = 1 << (k-1)
        dp[(mask, k)] = (dis[0][k], 0)
    for subset_size in range(2, n):
        from itertools import combinations
        for comb in combinations(range(1, n), subset_size):
            mask = 0
            for v in comb:
                mask |= 1 << (v-1)
            for j in comb:
                prev_mask = mask & ~(1 << (j-1))
                best_cost = math.inf
                best_prev = -1
                for k in comb:
                    if k == j: continue
                    if (prev_mask, k) in dp:
                        cost_k = dp[(prev_mask, k)][0] + dis[k][j]
                        if cost_k < best_cost:
                            best_cost = cost_k
                            best_prev = k
                dp[(mask, j)] = (best_cost, best_prev)
    full_mask = (1 << (n-1)) - 1
    best_cost = math.inf
    best_last = -1
    for k in range(1, n):
        if (full_mask, k) in dp:
            cost = dp[(full_mask, k)][0] + dis[k][0]
            if cost < best_cost:
                best_cost = cost
                best_last = k
    # reconstruct
    tour = [0]
    mask = full_mask
    last = best_last
    rev = []
    for _ in range(n-1):
        rev.append(last)
        prev = dp[(mask, last)][1]
        mask &= ~(1 << (last-1))
        last = prev
    tour.extend(reversed(rev))
    return best_cost, tour

# -------------------------
# Nearest Neighbor with 2-opt Local Search (IMPROVED VERSION)
# Phase 1: Greedy construction - start from a node, always go to the nearest unvisited node
# Phase 2: 2-opt refinement - systematically removes edge crossings and improves tour quality
# This hybrid approach provides 10-30% better tour lengths than basic NN while remaining fast.
# Still dependent on starting node, so best_nearest_neighbor tries multiple starts and picks the best.
# -------------------------
def two_opt_swap(tour: List[int], i: int, k: int) -> List[int]:
    """Reverse the tour segment between positions i and k (inclusive)."""
    new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
    return new_tour

def two_opt_improve(tour: List[int], dis: List[List[float]], max_iterations: int=1000) -> Tuple[float, List[int]]:
    """Apply 2-opt local search to improve a tour."""
    n = len(tour)
    improved = True
    iteration = 0
    best_tour = tour[:]
    best_length = evaluate_tour_by_index(best_tour, dis)

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, n - 1):
            for k in range(i + 1, n):
                # Calculate the change in tour length if we swap edges
                # Current edges: (tour[i-1], tour[i]) and (tour[k], tour[k+1 or 0])
                # New edges: (tour[i-1], tour[k]) and (tour[i], tour[k+1 or 0])

                edge1_old = dis[best_tour[i-1]][best_tour[i]]
                edge2_old = dis[best_tour[k]][best_tour[(k+1) % n]]
                edge1_new = dis[best_tour[i-1]][best_tour[k]]
                edge2_new = dis[best_tour[i]][best_tour[(k+1) % n]]

                delta = edge1_new + edge2_new - edge1_old - edge2_old

                if delta < -1e-10:  # Improvement found (with small tolerance for floating point)
                    new_tour = two_opt_swap(best_tour, i, k)
                    new_length = evaluate_tour_by_index(new_tour, dis)

                    if new_length < best_length:
                        best_tour = new_tour
                        best_length = new_length
                        improved = True
                        break  # Restart from beginning after improvement

            if improved:
                break

    return best_length, best_tour

def nearest_neighbor(points: Sequence[Point], start: int=0) -> Tuple[float, List[int]]:
    """Improved Nearest Neighbor: construct with NN, then refine with 2-opt."""
    # Phase 1: Construct initial tour with nearest neighbor
    n = len(points)
    dis = build_distance_matrix(points)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda v: dis[last][v])
        tour.append(nxt)
        unvisited.remove(nxt)

    # Phase 2: Improve tour with 2-opt local search
    improved_length, improved_tour = two_opt_improve(tour, dis)
    return improved_length, improved_tour

def best_nearest_neighbor(points: Sequence[Point], tries: int=8) -> Tuple[float, List[int], dict]:
    n = len(points)
    tries = min(tries, n)
    best_len = math.inf
    best_tour = None
    times = []
    for s in range(tries):
        t0 = time.perf_counter()
        l, t = nearest_neighbor(points, start=s)
        dt = time.perf_counter() - t0
        times.append(dt)
        if l < best_len:
            best_len = l
            best_tour = t
    return best_len, best_tour, {'tries':tries, 'times':times}

# -------------------------
# Ported GA (based on the JS functions you pasted)
# core ideas:
# - population: list of permutations (lists)
# - selection: keep a few elites + roulette selection based on 1/value fitness
# - crossover: getChild('next'/'previous') method reproduced
# - mutation: doMutate (reverse subsequence) and pushMutate (rotation of a block)
# - setBestValue: evaluate all and track best
# -------------------------
def random_individual(n:int) -> List[int]:
    a = list(range(n))
    random.shuffle(a)
    return a

def push_mutate(seq: List[int]) -> List[int]:
    # pushMutate: pick m,n where 0<=m<n<len, then s2 + s1 + s3
    L = len(seq)
    if L < 2: return seq[:]
    while True:
        m = random.randrange(L//2 + 1)  # original chooses from 0..len/2
        n = random.randrange(L)
        if m < n:
            break
    s1 = seq[:m]
    s2 = seq[m:n]
    s3 = seq[n:]
    return s2 + s1 + s3

def do_mutate(seq: List[int]) -> List[int]:
    # the JS doMutate reverses the subsequence between m and n (inclusive)
    L = len(seq)
    if L < 3: return seq[:]
    while True:
        m = random.randrange(L-2)
        n = random.randrange(L)
        if m < n:
            break
    new = seq[:]
    # reverse subsequence m..n
    i, j = m, n
    while i < j:
        new[i], new[j] = new[j], new[i]
        i += 1; j -= 1
    return new

def get_child(fun: str, px: List[int], py: List[int], dis: List[List[float]]) -> List[int]:
    # fun is 'next' or 'previous', we will use list.next/list.previous semantics
    # px, py are parent permutations (lists). Return solution list.
    n = len(px)
    px_copy = px[:]
    py_copy = py[:]
    sol = []
    # pick a random starting city (like c = px[randomIndex])
    c = px_copy[random.randrange(n)]
    sol.append(c)
    # while more than 1 left
    while len(px_copy) > 1:
        # dx = px.next(px.indexOf(c))
        # dy = py.next(py.indexOf(c))
        idx_px = px_copy.index(c)
        idx_py = py_copy.index(c)
        if fun == 'next':
            dx = px_copy[(idx_px+1) % len(px_copy)]
            dy = py_copy[(idx_py+1) % len(py_copy)]
        else:
            dx = px_copy[(idx_px-1) % len(px_copy)]
            dy = py_copy[(idx_py-1) % len(py_copy)]
        # delete c from both px_copy and py_copy
        px_copy.remove(c)
        py_copy.remove(c)
        # choose closer neighbour by dis
        # protect against missing entries if px_copy length==0
        if len(px_copy)==0:
            break
        # sometimes dx/dy might not be in px_copy/py_copy if removed; fallback to first
        if dx not in px_copy:
            dx = px_copy[0]
        if dy not in px_copy:
            dy = px_copy[0]
        # choose by comparing distances from c
        if dis[c][dx] < dis[c][dy]:
            c = dx
        else:
            c = dy
        sol.append(c)
    return sol

def do_crossover(pop: List[List[int]], x: int, y:int, dis:List[List[float]]):
    child1 = get_child('next', pop[x], pop[y], dis)
    child2 = get_child('previous', pop[x], pop[y], dis)
    pop[x] = child1
    pop[y] = child2

def set_roulette(values: List[float]) -> List[float]:
    # fitness = 1/value (lower value => higher fitness)
    fitness = [1.0/v if v>0 else 1e9 for v in values]
    total = sum(fitness)
    probs = [f/total for f in fitness]
    # cumulative
    for i in range(1, len(probs)):
        probs[i] += probs[i-1]
    return probs

def wheel_out(rand: float, roulette: List[float]) -> int:
    for i, threshold in enumerate(roulette):
        if rand <= threshold:
            return i
    return len(roulette)-1

def set_best_value(pop: List[List[int]], dis: List[List[float]]):
    values = [evaluate_tour_by_index(ind, dis) for ind in pop]
    bestpos = min(range(len(values)), key=lambda i: values[i])
    return {
        'values': values,
        'currentBest': {'bestPosition': bestpos, 'bestValue': values[bestpos]}
    }

def genetic_algorithm_points(points: Sequence[Point],
                             pop_size:int=30,
                             crossover_prob:float=0.9,
                             mutation_prob:float=0.01,
                             generations:int=200,
                             elite_keep:int=4,
                             seed:int=None,
                             trace:bool=False) -> Tuple[float, List[int], List[float]]:
    if seed is not None:
        random.seed(seed)
    n = len(points)
    dis = build_distance_matrix(points)
    # init population
    population = [random_individual(n) for _ in range(pop_size)]
    # evaluate
    meta = set_best_value(population, dis)
    values = meta['values']
    currentBest = meta['currentBest']
    best = population[currentBest['bestPosition']][:]
    bestValue = currentBest['bestValue']
    unchanged_gens = 0
    trace_best = [bestValue]
    # core loop
    for gen in range(generations):
        # selection: build new parents list. Start with some deterministic variants (like JS)
        parents = []
        # push best variants as in JS:
        parents.append(population[currentBest['bestPosition']])  # best
        parents.append(do_mutate(best[:]))  # mutated best
        parents.append(push_mutate(best[:]))  # pushed best
        parents.append(best[:])  # best clone
        # roulette selection for remaining slots
        # compute roulette
        roulette = set_roulette(values)
        while len(parents) < pop_size:
            idx = wheel_out(random.random(), roulette)
            parents.append(population[idx][:])
        population = parents
        # crossover: create queue of indices where crossover will happen
        queue = [i for i in range(pop_size) if random.random() < crossover_prob]
        random.shuffle(queue)
        for i in range(0, len(queue)-1, 2):
            do_crossover(population, queue[i], queue[i+1], dis)
        # mutation: randomly mutate some individuals
        for i in range(pop_size):
            if random.random() < mutation_prob:
                if random.random() > 0.5:
                    population[i] = push_mutate(population[i])
                else:
                    population[i] = do_mutate(population[i])
        # evaluate & update best
        meta = set_best_value(population, dis)
        values = meta['values']
        currentBest = meta['currentBest']
        if bestValue is None or currentBest['bestValue'] < bestValue:
            bestValue = currentBest['bestValue']
            best = population[currentBest['bestPosition']][:]
            unchanged_gens = 0
        else:
            unchanged_gens += 1
        if trace:
            trace_best.append(bestValue)
    return bestValue, best, trace_best

# -------------------------
# plotting helper (optional)
# -------------------------
def plot_tour(points: Sequence[Point], tour: Sequence[int], title: str, fname: str):
    if not HAS_PLT:
        return
    xs = [points[i][0] for i in tour] + [points[tour[0]][0]]
    ys = [points[i][1] for i in tour] + [points[tour[0]][1]]
    plt.figure(figsize=(6,6))
    plt.plot(xs, ys, marker='o')
    for i,p in enumerate(points):
        plt.text(p[0], p[1], str(i), fontsize=6)
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    os.makedirs('figures', exist_ok=True)
    plt.savefig(fname)
    plt.close()

# -------------------------
# Runner orchestration: run GA multiple times, NN, Held-Karp, record times
# -------------------------
def run_and_compare(points: Sequence[Point], instance_name: str, csv_writer, 
                    ga_params: dict, runs:int=13, hk_max_n:int=13):
    n = len(points)
    print(f"=== Instance {instance_name} n={n} ===")

    # -------------------------
    # Held-Karp multiple runs
    # -------------------------
    hk_lengths = []
    hk_times = []
    hk_tour = None
    if n <= hk_max_n:
        for run in range(runs):
            t0 = time.perf_counter()
            hk_len, hk_tour_run = held_karp(points)
            dt = time.perf_counter() - t0
            hk_lengths.append(hk_len)
            hk_times.append(dt)
            if hk_tour is None:
                hk_tour = hk_tour_run  # keep one for plotting
            csv_writer.writerow({
                'instance': instance_name, 'n': n, 'algorithm': 'Held-Karp', 'seed':'', 'run': run,
                'tour_length': hk_len, 'runtime_s': dt
            })
            print(f"Held-Karp run {run}: length={hk_len:.4f}, time={dt:.6f}s")
        print(f"Held-Karp summary: mean_length={statistics.mean(hk_lengths):.4f}, std={statistics.pstdev(hk_lengths):.4f}, best_length={min(hk_lengths):.4f}, mean_time={statistics.mean(hk_times):.6f}s")
    else:
        print("Held-Karp: skipped (n too large)")
        hk_len = None

    # -------------------------
    # Nearest Neighbor multiple runs
    # -------------------------
    nn_lengths = []
    nn_times = []
    nn_tours = []
    for run in range(runs):
        t0 = time.perf_counter()
        nn_len, nn_tour = nearest_neighbor(points, start=run % n)
        dt = time.perf_counter() - t0
        nn_lengths.append(nn_len)
        nn_times.append(dt)
        nn_tours.append(nn_tour)
        csv_writer.writerow({
            'instance': instance_name, 'n': n, 'algorithm': 'Nearest-Neighbor', 'seed':'', 'run': run,
            'tour_length': nn_len, 'runtime_s': dt
        })
        print(f"Nearest-NN run {run}: length={nn_len:.4f}, total_time={dt:.6f}s")
    print(f"Nearest-NN summary: mean_length={statistics.mean(nn_lengths):.4f}, std={statistics.pstdev(nn_lengths):.4f}, best_length={min(nn_lengths):.4f}, mean_time={statistics.mean(nn_times):.6f}s")

    # choose best NN tour for plotting
    best_nn_idx = nn_lengths.index(min(nn_lengths))
    best_nn_tour = nn_tours[best_nn_idx] if nn_tours else None

    # -------------------------
    # Genetic Algorithm multiple runs
    # -------------------------
    ga_lengths = []
    ga_times = []
    ga_traces = []
    ga_best_len = math.inf
    ga_best_tour = None
    for run in range(runs):
        seed = run
        t0 = time.perf_counter()
        g_len, g_tour, trace = genetic_algorithm_points(points, seed=seed, trace=True, **ga_params)
        dt = time.perf_counter() - t0
        ga_lengths.append(g_len)
        ga_times.append(dt)
        ga_traces.append(trace)
        if g_len < ga_best_len:
            ga_best_len = g_len
            ga_best_tour = g_tour
        csv_writer.writerow({
            'instance': instance_name, 'n': n, 'algorithm': 'Genetic Algorithm', 'seed': seed, 'run': run,
            'tour_length': g_len, 'runtime_s': dt
        })
        print(f"GA run {run}: length={g_len:.4f}, time={dt:.4f}s")
    print(f"GA summary: mean_length={statistics.mean(ga_lengths):.4f}, std={statistics.pstdev(ga_lengths):.4f}, best_length={ga_best_len:.4f}, mean_time={statistics.mean(ga_times):.4f}s")

    # -------------------------
    # Optional plotting of best tours (individual + combined)
    # -------------------------
    try:
        if HAS_PLT:
            os.makedirs('figures', exist_ok=True)

            # Individual: Held-Karp
            if hk_tour is not None:
                try:
                    plot_tour(points, hk_tour, f"Held-Karp opt (len={min(hk_lengths):.2f})", f"figures/{instance_name}_hk.png")
                except Exception as e:
                    print("Saving HK plot failed:", e)

            # Individual: Nearest-NN (best)
            if best_nn_tour is not None:
                try:
                    plot_tour(points, best_nn_tour, f"Nearest-NN best (len={min(nn_lengths):.2f})", f"figures/{instance_name}_nn_best.png")
                except Exception as e:
                    print("Saving NN best plot failed:", e)

            # Individual: GA best
            if ga_best_tour is not None:
                try:
                    plot_tour(points, ga_best_tour, f"GA best (len={ga_best_len:.2f})", f"figures/{instance_name}_ga_best.png")
                except Exception as e:
                    print("Saving GA best plot failed:", e)

            # Combined comparison (HK / NN / GA)
            try:
                plt.figure(figsize=(15,5))
                sub = 1
                total = 3
                if hk_tour is None:
                    total = 2  # no HK subplot if not available
                if hk_tour is not None:
                    plt.subplot(1,total,sub); sub += 1
                    xs = [points[i][0] for i in hk_tour] + [points[hk_tour[0]][0]]
                    ys = [points[i][1] for i in hk_tour] + [points[hk_tour[0]][1]]
                    plt.plot(xs, ys, marker='o'); plt.title('Held-Karp')
                    plt.gca().set_aspect('equal', adjustable='box')

                # Nearest-NN subplot
                plt.subplot(1,total,sub); sub += 1
                if best_nn_tour is not None:
                    xs = [points[i][0] for i in best_nn_tour] + [points[best_nn_tour[0]][0]]
                    ys = [points[i][1] for i in best_nn_tour] + [points[best_nn_tour[0]][1]]
                    plt.plot(xs, ys, marker='o')
                plt.title('Nearest-NN')
                plt.gca().set_aspect('equal', adjustable='box')

                # GA subplot
                plt.subplot(1,total,sub)
                if ga_best_tour is not None:
                    xs = [points[i][0] for i in ga_best_tour] + [points[ga_best_tour[0]][0]]
                    ys = [points[i][1] for i in ga_best_tour] + [points[ga_best_tour[0]][1]]
                    plt.plot(xs, ys, marker='o')
                plt.title('GA best')
                plt.gca().set_aspect('equal', adjustable='box')

                plt.suptitle(f"{instance_name} tours")
                plt.savefig(f"figures/{instance_name}_comparison.png")
                plt.close()
            except Exception as e:
                print("Saving combined comparison plot failed:", e)

    except Exception as e:
        print("Plotting failed:", e)


# -------------------------
# main: parse args, load points (from js or random), run experiments
# -------------------------
if __name__ == "__main__":
    import sys

    # Default paths
    default_js_file = os.path.join(os.path.dirname(__file__), "data.js")
    # change accordingly if needed, either data13, data 40, data200 or data 500
    default_dataset = "data13"

    # Parse args manually if run without command line args
    if len(sys.argv) == 1:
        # No arguments provided, use defaults
        js_data_file = default_js_file if os.path.exists(default_js_file) else None
        use_dataset = default_dataset
    else:
        # Use argparse if args are provided
        parser = argparse.ArgumentParser()
        parser.add_argument('--js-data-file', type=str, default=None, help='Path to data.js from the GitHub repo (optional)')
        parser.add_argument('--use', type=str, default='data200', help='Which dataset name to use when js-data-file provided, e.g. data40, data200')
        args = parser.parse_args()
        js_data_file = args.js_data_file
        use_dataset = args.use

    datasets = {}
    if js_data_file:
        try:
            datasets = load_points_from_js(js_data_file)
            print("Found datasets in js file:", list(datasets.keys()))
        except Exception as e:
            print("Failed to parse JS file:", e)
            datasets = {}

    if use_dataset not in datasets:
        print(f"Dataset {use_dataset} not found in JS file, available datasets:", list(datasets.keys()))
        sys.exit(1)

    points = datasets[use_dataset]
    # Prepare CSV
    csv_file = os.path.join(os.path.dirname(__file__), "tsp_results.csv")
    csv_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        fieldnames = ['instance','n','algorithm','seed','run','tour_length','runtime_s','hk_length','hk_time_s']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        # GA params
        ga_params = {'pop_size':30, 'crossover_prob':0.9, 'mutation_prob':0.01, 'generations':200, 'elite_keep':4}
        # Run experiment
        run_and_compare(points, use_dataset, writer, ga_params)

# length: total distance of the tour found in that particular run. 
# For Nearest Neighbor (NN), each run corresponds to a different starting node (or attempt), so its length can vary depending on the start.

# total_time (per run): time taken to compute that tour.

# mean_length: The arithmetic average of the tour lengths across all runs, tell us what a "Typical" result looks like.

# std_length (standard deviation): Measures how much the lengths vary across runs. 
# If std=0, all runs produced the same tour length, so algorithm is deterministic or highly consistent.
# If std>0, some runs are better than others.

# best_length: The shortest tour length found among all runs.

# mean_time: Average runtime over all runs.

# conclusion:
# For small n → exact algorithms (Held-Karp). For medium/large n → heuristics (GA, NN) and compare mean/best lengths versus runtime.
# Primary criterion is the least tour length found.
# GA will always give a better tour length than NN on average, But GA takes order of magnitude more CPU time per run.

# How to choose between GA and NN
# Use GA when any of these are true:
# You need better-quality tours (shorter lengths) and can afford the runtime.
# You only need to run the algorithm a small number of times (e.g., once per instance / offline computation).
# Your project / use-case values solution quality over latency.

# Prefer NN (or hybrid) when any of these are true:
# You need very fast solutions (real-time or many repeated queries).
# You must run the algorithm many times (e.g., in an online system, or running many experiments).
# You have strict CPU/time constraints.