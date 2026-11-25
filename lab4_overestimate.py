"""
Lab 04 - Task 3: what an overestimating heuristic does to A*.

1. Raises the heuristic of one node on the optimal route (Admin Block, D) far
   above its true remaining cost and shows the suboptimal answer, with the
   queue trace that explains it.
2. Sweeps the inflated value to find where optimality is lost.
3. Repeats the experiment on a node OFF the optimal route, where the same kind
   of overestimate is harmless, to show that one lucky run proves nothing.
4. Finds, for every node, the smallest overestimate that makes the route worse.
5. Runs a verification suite against a brute-force oracle.
"""

from lab4_astar import (GRAPH, H, START, GOAL, astar, path_cost, all_path_costs,
                        admissibility, consistency, true_costs_to_goal)

BAR = "=" * 78
BAD_D = 12          # inflated h(D); the true remaining cost h*(D) is 4


def arrow(path):
    return "-".join(path)


def with_h(node, value):
    h = dict(H)
    h[node] = value
    return h


def optimum():
    return min(all_path_costs(GRAPH, START, GOAL), key=lambda pc: pc[1])


def failing_run():
    star = true_costs_to_goal(GRAPH, GOAL)
    good = astar(GRAPH, H, START, GOAL)
    bad = astar(GRAPH, with_h("D", BAD_D), START, GOAL)
    best_path, best_cost = optimum()

    print(BAR)
    print(f"EXPERIMENT 1  inflate h(D) from {H['D']} to {BAD_D}  (true h*(D) = {star['D']})")
    print(BAR)
    print(f"{'configuration':<24}{'h(D)':>5}  {'path':<12}{'cost':>5}   verdict")
    print("-" * 64)
    for name, hd, res in (("admissible A*", H["D"], good), ("inadmissible A*", BAD_D, bad)):
        cost = path_cost(GRAPH, res.path)
        verdict = "optimal" if cost == best_cost else f"SUBOPTIMAL (+{cost - best_cost})"
        print(f"{name:<24}{hd:>5}  {arrow(res.path):<12}{cost:>5}   {verdict}")

    print("\nQueue trace of the inadmissible run (node popped, its f, queue afterwards)")
    print(f"{'step':<6}{'pop':<5}{'g':>3}{'f':>4}   queue after pop  node(f)")
    print("-" * 72)
    for step, node, g, h, f, path, queue, kind in bad.rows:
        q = "  ".join(f"{n}({ff})" for ff, gg, n in queue) or "-"
        print(f"{step:<6}{node:<5}{g:>3}{f:>4}   {q}{'   <- goal' if kind == 'goal' else ''}")


def sweep():
    best_path, best_cost = optimum()
    print("\n" + BAR)
    print("EXPERIMENT 2  how large must the overestimate be?")
    print(BAR)
    for node in ("D", "C"):
        star = true_costs_to_goal(GRAPH, GOAL)[node]
        on = node in best_path
        print(f"\nSweep h({node})  [h*({node}) = {star}, {node} is "
              f"{'ON' if on else 'OFF'} the optimal route]")
        print(f"{'h(' + node + ')':<8}{'path':<12}{'cost':>5}  verdict")
        print("-" * 40)
        last = None
        for value in range(star, star + 13):
            res = astar(GRAPH, with_h(node, value), START, GOAL)
            cost = path_cost(GRAPH, res.path)
            verdict = "optimal" if cost == best_cost else "SUBOPTIMAL"
            state = (arrow(res.path), cost)
            if state != last:                         # print only when result changes
                print(f"{value:<8}{arrow(res.path):<12}{cost:>5}  {verdict}")
                last = state
        print("(rows are shown only where the returned route changes)")


def per_node():
    """For each node: how much must h exceed h* before the route gets worse?"""
    best_path, best_cost = optimum()
    star = true_costs_to_goal(GRAPH, GOAL)
    print("\n" + BAR)
    print("EXPERIMENT 3  smallest harmful overestimate for each node")
    print(BAR)
    print(f"{'node':<6}{'h':>3}{'h*':>4}{'first bad h':>13}{'over by':>9}   route / cost")
    print("-" * 62)
    for node in sorted(GRAPH):
        if node == GOAL:
            continue
        hit = None
        for value in range(star[node] + 1, star[node] + 21):
            res = astar(GRAPH, with_h(node, value), START, GOAL)
            if path_cost(GRAPH, res.path) > best_cost:
                hit = (value, res)
                break
        if hit:
            value, res = hit
            print(f"{node:<6}{H[node]:>3}{star[node]:>4}{value:>13}{value - star[node]:>9}   "
                  f"{arrow(res.path)} / {path_cost(GRAPH, res.path)}")
        else:
            print(f"{node:<6}{H[node]:>3}{star[node]:>4}{'none':>13}{'-':>9}   "
                  f"optimal up to h* + 20")
    print(f"(optimal cost is {best_cost}; each node is inflated on its own)")


def verification():
    print("\n" + BAR)
    print("VERIFICATION SUITE")
    print(BAR)
    best_path, best_cost = optimum()
    star = true_costs_to_goal(GRAPH, GOAL)
    good = astar(GRAPH, H, START, GOAL)
    ucs = astar(GRAPH, {n: 0 for n in GRAPH}, START, GOAL)
    exact = astar(GRAPH, dict(star), START, GOAL)
    bad = astar(GRAPH, with_h("D", BAD_D), START, GOAL)
    off = astar(GRAPH, with_h("C", 40), START, GOAL)
    e_plus1 = astar(GRAPH, with_h("E", star["E"] + 1), START, GOAL)

    checks = [
        ("supplied heuristic is admissible", all(ok for *_, ok in admissibility(GRAPH, H, GOAL))),
        ("supplied heuristic is consistent", all(ok for *_, ok in consistency(GRAPH, H))),
        (f"brute-force optimum is {best_cost} min via {arrow(best_path)}", True),
        ("A* cost equals the brute-force optimum", path_cost(GRAPH, good.path) == best_cost),
        ("UCS cost equals the brute-force optimum", path_cost(GRAPH, ucs.path) == best_cost),
        ("A* expands fewer nodes than UCS", good.expanded_count < ucs.expanded_count),
        ("no node is expanded twice (consistent h)", len(set(good.expansions)) == good.expanded_count),
        ("exact heuristic expands only the optimal route",
         exact.expansions == best_path[:-1]),
        (f"h(D)={BAD_D} (overestimate) returns a worse route", path_cost(GRAPH, bad.path) > best_cost),
        ("the bad run never expands D", "D" not in bad.expansions),
        ("overestimating h(C) off the route stays optimal", path_cost(GRAPH, off.path) == best_cost),
        ("overestimating h(E) by only 1 minute already hurts",
         path_cost(GRAPH, e_plus1.path) > best_cost),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = sum(ok for _, ok in checks)
    print(f"\n{passed}/{len(checks)} checks passed")
    assert passed == len(checks), "a verification check failed"


if __name__ == "__main__":
    failing_run()
    sweep()
    per_node()
    verification()
