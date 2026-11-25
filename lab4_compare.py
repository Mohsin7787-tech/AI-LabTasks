

from lab4_astar import (GRAPH, H, START, GOAL, astar, path_cost,
                        true_costs_to_goal)

BAR = "=" * 78


def arrow(path):
    return "-".join(path)


def configs():
    star = true_costs_to_goal(GRAPH, GOAL)
    return [
        ("UCS (h = 0)", {n: 0 for n in GRAPH}),
        ("A* weak (h/2)", {n: H[n] // 2 for n in GRAPH}),
        ("A* supplied h", dict(H)),
        ("A* exact (h*)", {n: star[n] for n in GRAPH}),
    ]


def pop_table(title, h):
    res = astar(GRAPH, h, START, GOAL)
    print(f"\n{title}: pop order")
    print(f"{'step':<6}{'node':<6}{'g':>4}{'h':>4}{'f':>5}   path")
    print("-" * 50)
    for step, node, g, hn, f, path, _, kind in res.rows:
        tag = "  <- goal" if kind == "goal" else ""
        print(f"{step:<6}{node:<6}{g:>4}{hn:>4}{f:>5}   {arrow(path)}{tag}")


def main():
    print(BAR)
    print("A* VERSUS UCS  (expansions exclude the goal pop; start S, goal G)")
    print(BAR)
    print(f"{'configuration':<17}{'path':<14}{'cost':>5}  {'expanded':>8}  expansion order")
    print("-" * 78)
    results = {}
    for name, h in configs():
        res = astar(GRAPH, h, START, GOAL)
        results[name] = res
        print(f"{name:<17}{arrow(res.path):<14}{path_cost(GRAPH, res.path):>5}  "
              f"{res.expanded_count:>8}  {' '.join(res.expansions)}")
    print("-" * 78)

    ucs, sup = results["UCS (h = 0)"], results["A* supplied h"]
    skipped = [n for n in ucs.expansions if n not in sup.expansions]
    print(f"UCS expanded {ucs.expanded_count} nodes, supplied A* {sup.expanded_count}; "
          f"nodes A* never expanded: {' '.join(skipped)}")

    pop_table("UCS", dict(configs())["UCS (h = 0)"])
    pop_table("A* with the supplied heuristic", dict(H))


if __name__ == "__main__":
    main()
