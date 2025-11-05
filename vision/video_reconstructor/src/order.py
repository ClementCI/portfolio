import numpy as np
from tqdm import tqdm

# ==========================================================
#  Graph Ordering
# ==========================================================
def compute_length(order, dist):
    """
    Computes total distance of a path.
    """
    return sum(dist[order[i], order[i+1]] for i in range(len(order)-1))

def nn_heuristic(dist, start):
    """
    Nearest-neighbor TSP heuristic starting from 'start'.
    """
    n = len(dist)
    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    current = start
    
    for _ in range(n - 1):
        dists = dist[current]
        dists_masked = np.where(visited, np.inf, dists)
        nxt = int(np.argmin(dists_masked))
        order.append(nxt)
        visited[nxt] = True
        current = nxt
    
    return order

def two_opt(order, dist):
    """
    2-Opt refinement to improve NN TSP path.
    """
    improved = True
    n = len(order)
    while improved:
        improved = False
        for i in range(0, n - 1):
            for j in range(i + 1, n - 1):
                a, b = order[i - 1], order[i]
                c, d = order[j - 1], order[j]
                if dist[a, b] + dist[c, d] > dist[a, c] + dist[b, d]:
                    order[i:j] = order[i:j][::-1]
                    improved = True
    return order

def order_frames(embs, main_idx, n_restarts):
    """
    Temporally orders frames using NN + 2-opt TSP with random restarts.
    """
    main_embs = embs[main_idx]
    
    # Compute cosine similarity and convert to distance
    sim = main_embs @ main_embs.T
    np.fill_diagonal(sim, -np.inf)
    dist = 1.0 - sim
    dist = np.clip(dist, 0, 2)
    
    n = len(main_idx)
    best_order = None
    best_length = np.inf
    
    # Random restarts
    for _ in tqdm(range(n_restarts), total=n_restarts, desc="Solving TSP", unit="rd restart"):
        start = np.random.randint(0, n)      # random start
        order = nn_heuristic(dist, start)    # NN heuristic
        order = two_opt(order, dist)         # 2-opt refinement
        length = compute_length(order, dist)
        
        if length < best_length:
            best_order = order
            best_length = length
    
    # Map back to original frame indices
    ordered_frames = [main_idx[i] for i in best_order]
    return ordered_frames