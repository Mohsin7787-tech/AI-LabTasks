
#Lab 02 - Failure-mode analysis: sensor noise near a threshold.



from lab2_percepts import COOL_ABOVE, WARM_BELOW, reflex_decide
from lab2_model import StatefulAgent

DEADBAND = 0.5   
NOISY = [26.4, 25.8, 26.3, 25.7, 26.2, 25.9, 25.4, 24.8]


class HysteresisAgent:


    def __init__(self, start_mode="IDLE", deadband=DEADBAND):
        self.current_mode = start_mode
        self.deadband = deadband

    def _target(self, temp, occupied):
        if not occupied:
            return "ECO"
        if self.current_mode == "COOL":
            return "COOL" if temp > COOL_ABOVE - self.deadband else reflex_decide(temp, True)
        if self.current_mode == "WARM":
            return "WARM" if temp < WARM_BELOW + self.deadband else reflex_decide(temp, True)
        return reflex_decide(temp, True)

    def step(self, temp, occupied):
        target = self._target(temp, occupied)
        if target != self.current_mode:
            self.current_mode = target
            return target, "SEND"
        return target, "HOLD"


def run(agent, readings, occupied=True):
    log, sent = [], 0
    for t in readings:
        target, action = agent.step(t, occupied)
        sent += action == "SEND"
        log.append((t, target, action))
    return log, sent


def verification_suite():
    results = []

    def check(name, condition):
        results.append((name, bool(condition)))

    check("reflex: limits are strict (26.0 -> IDLE, 20.0 -> IDLE)",
          reflex_decide(26.0, True) == "IDLE" and reflex_decide(20.0, True) == "IDLE")
    check("reflex: empty room always ECO", reflex_decide(35.0, False) == "ECO")

    _, sent = run(StatefulAgent(), [28.0, 28.0, 28.0])
    check("stateful: repeated hot percept sends exactly 1 command", sent == 1)

    _, sent_state = run(StatefulAgent(), NOISY)
    _, sent_hyst = run(HysteresisAgent(), NOISY)
    check("stateful: chatters on noisy input (>= 5 commands)", sent_state >= 5)
    check("hysteresis: at most 2 commands on the same input", sent_hyst <= 2)

    cold, _ = run(HysteresisAgent(), [18.0, 18.4, 19.9])
    check("hysteresis: still enters and holds WARM when cold",
          [m for _, m, _ in cold] == ["WARM", "WARM", "WARM"])

    agent = HysteresisAgent("COOL")
    target, action = agent.step(30.0, False)
    check("hysteresis: occupancy override beats the deadband",
          target == "ECO" and action == "SEND")
    return results, sent_state, sent_hyst


def main():
    bar = "=" * 78
    print(bar)
    print("NOISE NEAR THE 26.0 C LIMIT - stateful agent vs hysteresis agent")
    print(f"Deadband = {DEADBAND} C  (leave COOL only at or below "
          f"{COOL_ABOVE - DEADBAND} C)")
    print(bar)

    s_log, s_sent = run(StatefulAgent(), NOISY)
    h_log, h_sent = run(HysteresisAgent(), NOISY)

    print(f"{'step':<6}{'temp':>6}   {'stateful':<16}{'hysteresis'}")
    print("-" * 56)
    for i, ((t, sm, sa), (_, hm, ha)) in enumerate(zip(s_log, h_log), start=1):
        print(f"{i:<6}{t:>6.1f}   {sm + ' ' + sa:<16}{hm} {ha}")
    print("-" * 56)
    print(f"Commands sent: stateful = {s_sent}, hysteresis = {h_sent} "
          f"(of {len(NOISY)} readings)")

    print("\nVERIFICATION SUITE")
    results, _, _ = verification_suite()
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} checks passed")
    assert passed == len(results), "a verification check failed"


if __name__ == "__main__":
    main()
