import heapq
from dataclasses import dataclass, field

START, GOAL = "S", "G"

NAMES = {"S": "Main Gate", "A": "Library", "B": "Cafeteria", "C": "Hostel",
         "D": "Admin Block", "E": "Lab Block", "F": "Sports Complex",
         "G": "Auditorium"}

EDGES = [
    ("S", "A", 2), ("S", "B", 3), ("S", "C", 5),
    ("A", "D", 4), ("A", "E", 6),
    ("B", "D", 2), ("B", "F", 4),
    ("C", "F", 2),
    ("D", "E", 1), ("D", "G", 5),
    ("E", "G", 3),
    ("F", "G", 6),
]

H = {"S": 8, "A": 6, "B": 5, "C": 6, "D": 3, "E": 2, "F": 4, "G": 0}


def build_graph(edges):
    graph = {}
    for u, v, w in edges:
        graph.setdefault(u, []).append((v, w))
        graph.setdefault(v, [])
    return graph


GRAPH = build_graph(EDGES)


@dataclass
class AStarResult:
    status: str
    path: list = None
    cost: float = None
    expansions: list = field(default_factory=list)   
    pops: list = field(default_factory=list)         
    rows: list = field(default_factory=list)       

    @property
    def expanded_count(self):
        return len(self.expansions)


def astar(graph, h, start, goal):
    best_g = {start: 0}
    heap = [(h[start], h[start], 0, 0, start, [start])]    
    order = 1
    res = AStarResult("FAILURE")
    while heap:
        f, hn, _, g, node, path = heapq.heappop(heap)
        if g > best_g.get(node, float("inf")):
            continue                                      
        res.pops.append(node)
        if node == goal:
            res.status, res.path, res.cost = "FOUND", path, g
            res.rows.append((len(res.pops), node, g, hn, f, path, _snapshot(heap, best_g), "goal"))
            return res
        res.expansions.append(node)
        for nxt, w in graph[node]:
            ng = g + w
            if ng < best_g.get(nxt, float("inf")):
                best_g[nxt] = ng
                heapq.heappush(heap, (ng + h[nxt], h[nxt], order, ng, nxt, path + [nxt]))
                order += 1
        res.rows.append((len(res.pops), node, g, hn, f, path, _snapshot(heap, best_g), "expand"))
    return res


def _snapshot(heap, best_g):
    """Valid queue entries, in the order they would be popped: (f, g, node)."""
    live = [e for e in heap if e[3] == best_g.get(e[4])]
    return [(e[0], e[3], e[4]) for e in sorted(live)]

def true_costs_to_goal(graph, goal):
    """h*(n): exact cheapest cost from every node to the goal, computed by
    running Dijkstra on the reversed graph."""
    rev = {n: [] for n in graph}
    for u, outs in graph.items():
        for v, w in outs:
            rev[v].append((u, w))
    dist = {goal: 0}
    heap = [(0, goal)]
    while heap:
        d, n = heapq.heappop(heap)
        if d > dist.get(n, float("inf")):
            continue
        for p, w in rev[n]:
            if d + w < dist.get(p, float("inf")):
                dist[p] = d + w
                heapq.heappush(heap, (d + w, p))
    return dist


def admissibility(graph, h, goal):
    star = true_costs_to_goal(graph, goal)
    return [(n, h[n], star.get(n), h[n] <= star.get(n, float("inf"))) for n in sorted(graph)]


def consistency(graph, h):
    rows = []
    for u in sorted(graph):
        for v, w in graph[u]:
            rows.append((u, v, w, h[u], h[v], h[u] <= w + h[v]))
    return rows


def all_path_costs(graph, start, goal):
    """Brute-force oracle: every simple path with its cost."""
    out = []

    def go(node, path, cost):
        if node == goal:
            out.append((list(path), cost))
            return
        for nxt, w in graph[node]:
            if nxt not in path:
                go(nxt, path + [nxt], cost + w)

    go(start, [start], 0)
    return out


def path_cost(graph, path):
    lookup = {(u, v): w for u, outs in graph.items() for v, w in outs}
    return sum(lookup[(a, b)] for a, b in zip(path, path[1:]))
