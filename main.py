from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import heapq
import random
import math
import time
from collections import deque

app = FastAPI(title="AI Snake Pathfinding API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ───────────────────────────────────────────────────────────────────

class GameState(BaseModel):
    snake: List[List[int]]        # [[row, col], ...]
    food: List[int]               # [row, col]
    obstacles: List[List[int]]    # [[row, col], ...]
    grid_size: int
    algorithm: Optional[str] = None   # force specific algo
    scenario: Optional[str] = None

class MoveResponse(BaseModel):
    direction: List[int]          # [dr, dc]
    algorithm: str
    path_length: int              # How many steps to food
    steps_evaluated: int          # Cells examined during search
    explanation: str              # Human-friendly reasoning
    alternatives: List[Dict[str, Any]]  # 2 alternative strategies

# ─── Helpers ──────────────────────────────────────────────────────────────────

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def is_valid(r, c, grid_size, obstacles, snake_set):
    return (0 <= r < grid_size and 0 <= c < grid_size
            and (r, c) not in obstacles
            and (r, c) not in snake_set)

def neighbors(pos, grid_size, obstacles, snake_set):
    r, c = pos
    result = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if is_valid(nr, nc, grid_size, obstacles, snake_set):
            result.append((nr, nc))
    return result

def reconstruct(came_from, start, goal):
    path = []
    cur = goal
    while cur != start:
        path.append(list(cur))
        cur = came_from[cur]
    path.reverse()
    return path

# ─── Algorithms ───────────────────────────────────────────────────────────────

def astar(snake, food, obstacles, grid_size) -> Dict:
    """A* search - optimal path using heuristic + actual cost"""
    start = tuple(snake[0])
    goal = tuple(food)
    snake_set = {tuple(s) for s in snake}
    obs_set = {tuple(o) for o in obstacles}

    open_heap = [(manhattan(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    explored_nodes = 0  # renamed from 'steps' for clarity

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        explored_nodes += 1
        if cur == goal:
            path = reconstruct(came_from, start, goal)
            return {"path": path, "steps": explored_nodes, "success": True}
        if g > g_score.get(cur, float('inf')):
            continue
        for nb in neighbors(cur, grid_size, obs_set, snake_set - {start}):
            ng = g + 1
            if ng < g_score.get(nb, float('inf')):
                g_score[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + manhattan(nb, goal), ng, nb))

    return {"path": [], "steps": explored_nodes, "success": False}

def bfs(snake, food, obstacles, grid_size) -> Dict:
    """Breadth-first search - guarantees shortest path in unweighted grid"""
    start = tuple(snake[0])
    goal = tuple(food)
    snake_set = {tuple(s) for s in snake}
    obs_set = {tuple(o) for o in obstacles}

    queue = deque([start])
    came_from = {start: None}
    explored_nodes = 0

    while queue:
        cur = queue.popleft()
        explored_nodes += 1
        if cur == goal:
            path = []
            c = cur
            while came_from[c] is not None:
                path.append(list(c))
                c = came_from[c]
            path.reverse()
            return {"path": path, "steps": explored_nodes, "success": True}
        for nb in neighbors(cur, grid_size, obs_set, snake_set - {start}):
            if nb not in came_from:
                came_from[nb] = cur
                queue.append(nb)

    return {"path": [], "steps": explored_nodes, "success": False}

def dfs(snake, food, obstacles, grid_size) -> Dict:
    """Depth-first search - explores deep paths first, good for tunnels"""
    start = tuple(snake[0])
    goal = tuple(food)
    snake_set = {tuple(s) for s in snake}
    obs_set = {tuple(o) for o in obstacles}
    explored_nodes = [0]

    def _dfs(cur, visited, path):
        explored_nodes[0] += 1
        if cur == goal:
            return list(path)
        if explored_nodes[0] > grid_size * grid_size * 2:
            return None
        for nb in neighbors(cur, grid_size, obs_set, snake_set - {start}):
            if nb not in visited:
                visited.add(nb)
                path.append(list(nb))
                result = _dfs(nb, visited, path)
                if result is not None:
                    return result
                path.pop()
        return None

    visited = {start}
    result = _dfs(start, visited, [])
    return {"path": result or [], "steps": explored_nodes[0], "success": result is not None}

def ucs(snake, food, obstacles, grid_size) -> Dict:
    """Uniform Cost Search — avoids risky areas near the snake body."""
    start = tuple(snake[0])
    goal = tuple(food)
    snake_set = {tuple(s) for s in snake}
    obs_set = {tuple(o) for o in obstacles}

    def cell_cost(r, c):
        base = 1
        # Penalize cells near snake body (danger zones)
        for i, (sr, sc) in enumerate(snake[1:], 1):
            d = abs(r - sr) + abs(c - sc)
            if d == 1:
                base += max(0, 5 - i)
        return base

    open_heap = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}
    explored_nodes = 0

    while open_heap:
        cost, cur = heapq.heappop(open_heap)
        explored_nodes += 1
        if cur == goal:
            path = reconstruct(came_from, start, goal)
            return {"path": path, "steps": explored_nodes, "success": True}
        if cost > cost_so_far.get(cur, float('inf')):
            continue
        for nb in neighbors(cur, grid_size, obs_set, snake_set - {start}):
            new_cost = cost_so_far[cur] + cell_cost(*nb)
            if new_cost < cost_so_far.get(nb, float('inf')):
                cost_so_far[nb] = new_cost
                came_from[nb] = cur
                heapq.heappush(open_heap, (new_cost, nb))

    return {"path": [], "steps": explored_nodes, "success": False}

def greedy(snake, food, obstacles, grid_size) -> Dict:
    """Greedy best-first - charges toward food using only heuristic"""
    start = tuple(snake[0])
    goal = tuple(food)
    snake_set = {tuple(s) for s in snake}
    obs_set = {tuple(o) for o in obstacles}

    open_heap = [(manhattan(start, goal), start)]
    came_from = {start: None}
    visited = {start}
    explored_nodes = 0

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        explored_nodes += 1
        if cur == goal:
            path = []
            c = cur
            while came_from[c] is not None:
                path.append(list(c))
                c = came_from[c]
            path.reverse()
            return {"path": path, "steps": explored_nodes, "success": True}
        for nb in neighbors(cur, grid_size, obs_set, snake_set - {start}):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = cur
                heapq.heappush(open_heap, (manhattan(nb, goal), nb))

    return {"path": [], "steps": explored_nodes, "success": False}

def hill_climbing(snake, food, obstacles, grid_size) -> Dict:
    """Hill climbing - always moves to best neighbor, can get stuck"""
    start = tuple(snake[0])
    goal = tuple(food)
    snake_set = {tuple(s) for s in snake}
    obs_set = {tuple(o) for o in obstacles}

    cur = start
    path = []
    visited = {start}
    explored_nodes = 0

    for _ in range(grid_size * grid_size):
        explored_nodes += 1
        if cur == goal:
            return {"path": path, "steps": explored_nodes, "success": True}
        nbs = neighbors(cur, grid_size, obs_set, snake_set - {start})
        nbs = [n for n in nbs if n not in visited]
        if not nbs:
            break
        best = min(nbs, key=lambda n: manhattan(n, goal))
        if manhattan(best, goal) >= manhattan(cur, goal):
            break  # local minimum — stuck
        visited.add(best)
        path.append(list(best))
        cur = best

    return {"path": path, "steps": explored_nodes, "success": cur == goal}

def simulated_annealing(snake, food, obstacles, grid_size) -> Dict:
    """Simulated annealing - sometimes takes worse moves to escape traps"""
    start = tuple(snake[0])
    goal = tuple(food)
    snake_set = {tuple(s) for s in snake}
    obs_set = {tuple(o) for o in obstacles}

    cur = start
    path = [list(start)]
    visited = {start}
    temp = 10.0
    cooling = 0.95
    explored_nodes = 0

    for _ in range(grid_size * grid_size * 3):
        explored_nodes += 1
        if cur == goal:
            return {"path": path[1:], "steps": explored_nodes, "success": True}
        nbs = neighbors(cur, grid_size, obs_set, snake_set - {start})
        if not nbs:
            break
        nxt = random.choice(nbs)
        delta = manhattan(cur, goal) - manhattan(nxt, goal)
        if delta > 0 or (temp > 0.1 and random.random() < math.exp(delta / temp)):
            path.append(list(nxt))
            visited.add(nxt)
            cur = nxt
        temp *= cooling

    return {"path": path[1:], "steps": explored_nodes, "success": cur == goal}

def ids(snake, food, obstacles, grid_size) -> Dict:
    """Iterative Deepening Search - memory-efficient deep search"""
    start = tuple(snake[0])
    goal = tuple(food)
    snake_set = {tuple(s) for s in snake}
    obs_set = {tuple(o) for o in obstacles}
    total_explored = [0]

    def dls(cur, depth, path, visited):
        total_explored[0] += 1
        if cur == goal:
            return list(path)
        if depth == 0:
            return None
        for nb in neighbors(cur, grid_size, obs_set, snake_set - {start}):
            if nb not in visited:
                visited.add(nb)
                path.append(list(nb))
                result = dls(nb, depth - 1, path, visited)
                if result is not None:
                    return result
                path.pop()
                visited.discard(nb)
        return None

    for max_depth in range(1, grid_size * 2):
        result = dls(start, max_depth, [], {start})
        if result is not None:
            return {"path": result, "steps": total_explored[0], "success": True}
        if total_explored[0] > grid_size * grid_size * 4:
            break

    return {"path": [], "steps": total_explored[0], "success": False}

def bidirectional(snake, food, obstacles, grid_size) -> Dict:
    """Bidirectional search - meets in the middle, great for large grids"""
    start = tuple(snake[0])
    goal = tuple(food)
    snake_set = {tuple(s) for s in snake}
    obs_set = {tuple(o) for o in obstacles}

    if start == goal:
        return {"path": [], "steps": 0, "success": True}

    fwd = {start: None}
    bwd = {goal: None}
    fwd_q = deque([start])
    bwd_q = deque([goal])
    explored_nodes = 0

    def trace(node, came_from_dict):
        path = []
        c = node
        while came_from_dict[c] is not None:
            path.append(list(c))
            c = came_from_dict[c]
        return path

    while fwd_q or bwd_q:
        # Expand forward
        if fwd_q:
            cur = fwd_q.popleft()
            explored_nodes += 1
            for nb in neighbors(cur, grid_size, obs_set, snake_set - {start}):
                if nb not in fwd:
                    fwd[nb] = cur
                    fwd_q.append(nb)
                    if nb in bwd:
                        p1 = trace(nb, fwd)
                        p1.reverse()
                        p2 = trace(nb, bwd)
                        return {"path": p1 + p2, "steps": explored_nodes, "success": True}

        # Expand backward
        if bwd_q:
            cur = bwd_q.popleft()
            explored_nodes += 1
            for nb in neighbors(cur, grid_size, obs_set, snake_set - {start}):
                if nb not in bwd:
                    bwd[nb] = cur
                    bwd_q.append(nb)
                    if nb in fwd:
                        p1 = trace(nb, fwd)
                        p1.reverse()
                        p2 = trace(nb, bwd)
                        return {"path": p1 + p2, "steps": explored_nodes, "success": True}

    return {"path": [], "steps": explored_nodes, "success": False}

# ─── Algorithm Registry ───────────────────────────────────────────────────────

ALGORITHMS = {
    "astar":      (astar,      "A*",              "Optimal path", "Balances distance and exploration — finds the shortest route efficiently."),
    "bfs":        (bfs,        "BFS",             "Shortest path", "Explores outward like ripples in water — guarantees the fewest steps."),
    "dfs":        (dfs,        "DFS",             "Deep explorer", "Dives deep into corridors — great for narrow tunnels and mazes."),
    "ucs":        (ucs,        "UCS",             "Risk-aware", "Avoids risky areas near the snake body — safer but sometimes longer."),
    "greedy":     (greedy,     "Greedy BFS",      "Fast and direct", "Charges straight toward food — quick but might miss the best path."),
    "hill":       (hill_climbing, "Hill Climbing", "Local optimizer", "Always moves to the closest neighbor — fast but can get trapped."),
    "sa":         (simulated_annealing, "Simulated Annealing", "Escape artist", "Sometimes takes worse moves to escape traps — unpredictable but clever."),
    "ids":        (ids,        "IDS",             "Memory saver", "Repeatedly deep-dives with increasing limits — thorough but methodical."),
    "bidir":      (bidirectional, "Bidirectional", "Two-front attack", "Searches from both snake and food — meets in the middle for speed."),
}

# ─── Scenario → Algorithm Mapping ─────────────────────────────────────────────

SCENARIO_INFO = {
    "maze": {
        "best": "bfs",
        "why": "BFS guarantees the shortest path through winding corridors — no wrong turns.",
        "description": "Dense maze — narrow corridors, many dead ends",
        "obstacle_style": "maze"
    },
    "open": {
        "best": "bidir",
        "why": "Bidirectional search finds the path faster by working from both ends.",
        "description": "Open field — minimal obstacles, large space",
        "obstacle_style": "sparse"
    },
    "danger": {
        "best": "ucs",
        "why": "UCS steers clear of the snake's body — safer navigation through danger zones.",
        "description": "Danger zones — high-cost cells near the snake tail",
        "obstacle_style": "clustered"
    },
    "deep": {
        "best": "dfs",
        "why": "DFS commits to deep corridors and backtracks when needed — perfect for tunnels.",
        "description": "Deep tunnels — long, narrow passages",
        "obstacle_style": "tunnels"
    },
    "trap": {
        "best": "sa",
        "why": "Simulated Annealing's random jumps escape traps that fool greedy algorithms.",
        "description": "Trap layout — pockets that lure greedy algorithms",
        "obstacle_style": "traps"
    },
    "weighted": {
        "best": "ucs",
        "why": "UCS handles variable costs naturally — optimal for weighted cells.",
        "description": "Weighted grid — variable movement costs",
        "obstacle_style": "weighted"
    },
    "large": {
        "best": "bidir",
        "why": "Bidirectional search explores much less space on huge grids.",
        "description": "Large grid — maximum search space",
        "obstacle_style": "scattered"
    },
    "optimal": {
        "best": "astar",
        "why": "A* finds the perfect balance — shortest path without wasted exploration.",
        "description": "Clean grid — optimal pathfinding showcase",
        "obstacle_style": "moderate"
    },
}

# ─── Helper: Generate human-friendly explanation ─────────────────────────────

def get_explanation(name: str, success: bool, elapsed_ms: float, steps: int, path_len: int):
    """Convert technical metrics into friendly explanations."""
    if not success or path_len == 0:
        return "No clear path to the food — taking a safe move to stay alive"
    
    # Add variety to speed feedback
    if elapsed_ms < 2.0:
        speed_note = random.choice([
            "Found a clear path quickly",
            "Quick route found",
            "Fast path detection"
        ])
    elif elapsed_ms < 5.0:
        speed_note = random.choice([
            "Took a bit longer due to obstacles",
            "Obstacles slowed things down a bit",
            "Found a path despite some blockers"
        ])
    else:
        speed_note = random.choice([
            "Struggled to find a safe route",
            "The grid is tricky here",
            "Had to work hard to find a path"
        ])
    
    # Path quality with natural phrasing
    if path_len <= 5:
        quality = random.choice([
            "Food is very close!",
            "The route to food looks safe",
            "A quick snack is nearby"
        ])
    elif path_len <= 15:
        quality = random.choice([
            "Found a reliable path",
            "A decent route to the food",
            "The path looks clear enough"
        ])
    else:
        quality = random.choice([
            "Long path ahead, but it's safe",
            "A winding route keeps us alive",
            "Taking the scenic route, but it works"
        ])
    
    return f"{speed_note}. {quality}"

# ─── Endpoint: Get Move ───────────────────────────────────────────────────────

@app.post("/move", response_model=MoveResponse)
def get_move(state: GameState):
    head = state.snake[0]
    food = state.food
    obstacles = state.obstacles
    grid_size = state.grid_size
    scenario = state.scenario or "optimal"

    # Choose algorithm based on scenario or user override
    scenario_data = SCENARIO_INFO.get(scenario, SCENARIO_INFO["optimal"])
    algo_key = state.algorithm or scenario_data["best"]
    if algo_key not in ALGORITHMS:
        algo_key = "astar"

    fn, name, strategy, description = ALGORITHMS[algo_key]
    t0 = time.perf_counter()
    result = fn(state.snake, food, obstacles, grid_size)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    path = result["path"]
    explored_nodes = result["steps"]  # renamed internally for clarity
    success = result["success"]

    # Determine movement direction from path or use safe fallback
    direction = [0, 0]
    if path:
        nr, nc = path[0]
        direction = [nr - head[0], nc - head[1]]
    else:
        # No path found - find any safe adjacent cell
        snake_set = {tuple(s) for s in state.snake}
        obs_set = {tuple(o) for o in obstacles}
        for dr, dc in DIRS:
            nr, nc = head[0] + dr, head[1] + dc
            if is_valid(nr, nc, grid_size, obs_set, snake_set):
                direction = [dr, dc]
                break

    # Run 2 alternative algorithms for comparison with helpful notes
    alts = []
    alt_keys = [k for k in ["astar", "bfs", "ucs", "greedy", "bidir"] if k != algo_key][:2]
    
    # Simple notes for alternatives
    alt_notes = {
        "astar": "Balanced & reliable",
        "bfs": "Explores more nodes",
        "ucs": "Safer but longer",
        "greedy": "Fast but risky",
        "bidir": "Great for open spaces"
    }
    
    for ak in alt_keys:
        afn, aname, astrategy, _ = ALGORITHMS[ak]
        ares = afn(state.snake, food, obstacles, grid_size)
        alts.append({
            "algorithm": aname,
            "path_length": len(ares["path"]),
            "success": ares["success"],
            "note": alt_notes.get(ak, "Standard approach")
        })

    return MoveResponse(
        direction=direction,
        algorithm=name,
        path_length=len(path),
        steps_evaluated=explored_nodes,
        explanation=get_explanation(name, success, elapsed_ms, explored_nodes, len(path)),
        alternatives=alts,
    )


@app.get("/scenarios")
def get_scenarios():
    return {k: {"description": v["description"], "best": v["best"], "why": v["why"]}
            for k, v in SCENARIO_INFO.items()}


@app.get("/algorithms")
def get_algorithms():
    return {k: {"name": v[1], "strategy": v[2], "description": v[3]}
            for k, v in ALGORITHMS.items()}


@app.get("/health")
def health():
    return {"status": "ok"}