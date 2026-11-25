from lab4_astar import (GRAPH, H, START, GOAL, NAMES, astar, admissibility,
                        consistency, all_path_costs)

BAR = "=" * 78


def part_a():
    print(BAR)
    print("PART A  HEURISTIC AUDIT")
    print(BAR)
    print("\nAdmissibility: h(n) must not exceed the true remaining cost h*(n)")
    print(f"{'node':<6}{'place':<18}{'h(n)':>6}{'h*(n)':>7}   check")
    print("-" * 52)
    for n, h, star, ok in admissibility(GRAPH, H, GOAL):
        print(f"{n:<6}{NAMES[n]:<18}{h:>6}{star:>7}   {h} <= {star}  {'ok' if ok else 'VIOLATED'}")

    print("\nConsistency: h(u) <= cost(u,v) + h(v) on every directed link")
    print(f"{'link':<8}{'cost':>5}{'h(u)':>6}{'h(v)':>6}   check")
    print("-" * 52)
    for u, v, w, hu, hv, ok in consistency(GRAPH, H):
        print(f"{u}->{v:<5}{w:>5}{hu:>6}{hv:>6}   {hu} <= {w} + {hv} = {w + hv}  "
              f"{'ok' if ok else 'VIOLATED'}")


def part_b():
    print("\n" + BAR)
    print("PART B  FIRST THREE EXPANSIONS  (f = g + h, ties: smaller h, then FIFO)")
    print(BAR)
    res = astar(GRAPH, H, START, GOAL)
    for step, node, g, h, f, path, queue, kind in res.rows[:3]:
        print(f"\nStep {step}: pop {node} ({NAMES[node]})   g={g}  h={h}  f={g}+{h}={f}")
        for nxt, w in GRAPH[node]:
            print(f"    successor {nxt}: g={g}+{w}={g + w}, h={H[nxt]}, f={g + w + H[nxt]}")
        print("    queue after: " + "  ".join(f"{n}(f={ff},g={gg})" for ff, gg, n in queue))
    nxt = res.rows[3]
    print(f"\nNext pop would be {nxt[1]} with f={nxt[4]}.")


def main():
    part_a()
    part_b()
    best = min(c for _, c in all_path_costs(GRAPH, START, GOAL))
    print(f"\nBrute-force optimum from {START} to {GOAL}: {best} minutes")


if __name__ == "__main__":
    main()
