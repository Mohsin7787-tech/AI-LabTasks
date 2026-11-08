
#Lab 01 - Task 2: Symbolic (rule-based) scholarship screening.

import csv
from dataclasses import dataclass

MIN_SCORE = 70.0   
MIN_GPA = 3.0      


@dataclass(frozen=True)
class Applicant:
    score: float
    documents_complete: bool
    prerequisite_gpa: float
    valid_recommendation: bool


@dataclass(frozen=True)
class Decision:
    label: str           
    passed: tuple         
    failed: tuple         

    def explain(self):
        if not self.failed:
            return "all four checks satisfied"
        return "failed: " + ", ".join(self.failed)


RULES = [
    ("documents_complete", lambda a: a.documents_complete is True),
    ("score>=70", lambda a: a.score >= MIN_SCORE),
    ("gpa>=3.0", lambda a: a.prerequisite_gpa >= MIN_GPA),
    ("recommendation", lambda a: a.valid_recommendation is True),
]


def evaluate(applicant):
    """Apply every rule and return a Decision with a full audit trail."""
    passed, failed = [], []
    for name, check in RULES:
        (passed if check(applicant) else failed).append(name)
    label = "Review" if not failed else "Hold"
    return Decision(label, tuple(passed), tuple(failed))


TEST_CASES = [
    ("C1 standard pass",
     Applicant(88.0, True, 3.6, True), "Review",
     "baseline: every condition met"),
    ("C2 GPA failure",
     Applicant(94.0, True, 2.7, True), "Hold",
     "very high score cannot compensate for low GPA"),
    ("C3 no recommendation",
     Applicant(79.0, True, 3.3, False), "Hold",
     "strong profile blocked by missing endorsement"),
    ("C4 exact boundary",
     Applicant(70.0, True, 3.0, True), "Review",
     "score and GPA sit exactly on their thresholds"),
    ("C5 documents missing",
     Applicant(81.0, False, 3.4, True), "Hold",
     "paperwork gate blocks an otherwise good applicant"),
]


def run():
    bar = "=" * 78
    print(bar)
    print("SCHOLARSHIP SCREENING - SYMBOLIC RULE ENGINE (Lab 01, Task 2)")
    print(f"Thresholds are synthetic lab assumptions: score >= {MIN_SCORE}, GPA >= {MIN_GPA}")
    print(bar)

    rows, correct = [], 0
    for name, app, expected, why in TEST_CASES:
        decision = evaluate(app)
        ok = decision.label == expected
        correct += ok
        print(f"\n{name}")
        print(f"  input    : score={app.score}, docs={app.documents_complete}, "
              f"gpa={app.prerequisite_gpa}, rec={app.valid_recommendation}")
        print(f"  expected : {expected}")
        print(f"  actual   : {decision.label}   [{'PASS' if ok else 'FAIL'}]")
        print(f"  audit    : {decision.explain()}")
        print(f"  purpose  : {why}")
        rows.append({"case": name, "score": app.score,
                     "docs": app.documents_complete, "gpa": app.prerequisite_gpa,
                     "rec": app.valid_recommendation, "expected": expected,
                     "actual": decision.label, "failed_checks": ";".join(decision.failed)})

    print("\n" + bar)
    print(f"Summary: {correct}/{len(TEST_CASES)} test cases matched the expected label.")
    print(bar)

    with open("rules_results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote rules_results.csv")
    assert correct == len(TEST_CASES), "A rule-engine test case failed"


if __name__ == "__main__":
    run()
