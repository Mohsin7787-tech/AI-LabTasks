
#Lab 01 - Paradigm comparison: symbolic rules vs a probability threshold.


import csv
from lab1_rules import Applicant, evaluate

PROB_CUTOFF = 0.70   

SCENARIOS = [
    ("complete_high",   84.0, True,  0.83),
    ("complete_low",    66.0, True,  0.72),
    ("incomplete_high", 93.0, False, 0.90),
    ("boundary",        70.0, True,  0.70),
    ("weak_all_round",  55.0, True,  0.41),
]


def symbolic_label(score, docs):
    return evaluate(Applicant(score, docs, 3.5, True)).label


def probabilistic_label(prob):
    return "Review" if prob >= PROB_CUTOFF else "Hold"


def compare():
    rows = []
    for name, score, docs, prob in SCENARIOS:
        sym, pro = symbolic_label(score, docs), probabilistic_label(prob)
        rows.append({"scenario": name, "score": score, "docs": docs, "prob": prob,
                     "symbolic": sym, "probabilistic": pro, "disagree": sym != pro})
    return rows


def sensitivity():
    """One-step perturbations around the boundary for each paradigm."""
    out = []
    for score in (71.0, 70.0, 69.0):
        out.append(("symbolic score", score, symbolic_label(score, True)))
    for prob in (0.71, 0.70, 0.69):
        out.append(("probability", prob, probabilistic_label(prob)))
    return out


def main():
    bar = "=" * 86
    print(bar)
    print("PARADIGM COMPARISON - SYMBOLIC RULES vs PROBABILITY THRESHOLD")
    print(f"Probability cut-off = {PROB_CUTOFF} (synthetic assumption)")
    print(bar)

    rows = compare()
    head = f"{'scenario':<16}{'score':>6}{'docs':>7}{'prob':>7}  {'symbolic':<9}{'prob-model':<11}{'disagree'}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['scenario']:<16}{r['score']:>6.0f}{str(r['docs']):>7}{r['prob']:>7.2f}  "
              f"{r['symbolic']:<9}{r['probabilistic']:<11}{r['disagree']}")

    total = sum(r["disagree"] for r in rows)
    risky = [r["scenario"] for r in rows
             if r["probabilistic"] == "Review" and r["symbolic"] == "Hold"]
    print("-" * len(head))
    print(f"Disagreements: {total} of {len(rows)} scenarios")
    print(f"Probability model says Review where rules say Hold: {', '.join(risky) or 'none'}")

    print("\nSENSITIVITY AROUND THE BOUNDARY")
    print(f"{'paradigm':<16}{'input':>8}  outcome")
    for paradigm, value, outcome in sensitivity():
        shown = f"{value:.2f}" if paradigm == "probability" else f"{value:.0f}"
        print(f"{paradigm:<16}{shown:>8}  {outcome}")

    with open("compare_results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nWrote compare_results.csv")


if __name__ == "__main__":
    main()
