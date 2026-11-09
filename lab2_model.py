#Lab 02 - Task 3 (part B): model-based (stateful) agent.


from lab2_percepts import PERCEPTS, reflex_decide, COOL_ABOVE, WARM_BELOW


class StatefulAgent:
    def __init__(self, start_mode="IDLE"):
        self.current_mode = start_mode          

    def step(self, temp, occupied):
        """Return (target_mode, 'SEND' or 'HOLD')."""
        target = reflex_decide(temp, occupied)
        if target != self.current_mode:
            self.current_mode = target
            return target, "SEND"
        return target, "HOLD"


def main():
    print("=" * 74)
    print("STATEFUL (MODEL-BASED) AGENT - classroom energy controller")
    print(f"Limits (synthetic): cool above {COOL_ABOVE}C, warm below {WARM_BELOW}C")
    print("=" * 74)
    print(f"{'step':<6}{'was':<7}{'temp':>6} {'occ':<7}{'target':<8}{'action':<8}note")
    print("-" * 74)

    agent, sent = StatefulAgent(), 0
    for step, (temp, occ) in enumerate(PERCEPTS, start=1):
        was = agent.current_mode
        target, action = agent.step(temp, occ)
        sent += action == "SEND"
        note = f"{was} -> {target}" if action == "SEND" else "already in this mode"
        print(f"{step:<6}{was:<7}{temp:>6.1f} {str(occ):<7}{target:<8}{action:<8}{note}")

    print("-" * 74)
    print(f"Commands transmitted: {sent} of {len(PERCEPTS)} steps "
          f"(reflex agent sends {len(PERCEPTS)})")


if __name__ == "__main__":
    main()
