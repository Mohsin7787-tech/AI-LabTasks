
#Lab 02 - Task 3 (part A): simple reflex agent for a smart-classroom controller.



COOL_ABOVE = 26.0   
WARM_BELOW = 20.0   


PERCEPTS = [
    (28.5, True),    
    (28.5, True),    
    (26.0, True),    
    (20.0, True),    
    (18.5, True),  
    (28.5, False),   
]


def reflex_decide(temp, occupied):
    
    if not occupied:
        return "ECO"
    if temp > COOL_ABOVE:
        return "COOL"
    if temp < WARM_BELOW:
        return "WARM"
    return "IDLE"


def main():
    print("=" * 66)
    print("SIMPLE REFLEX AGENT - classroom energy controller")
    print(f"Limits (synthetic): cool above {COOL_ABOVE}C, warm below {WARM_BELOW}C")
    print("=" * 66)
    print(f"{'step':<6}{'temp':>7}  {'occupied':<10}{'mode chosen':<12}{'command sent'}")
    print("-" * 66)

    commands = 0
    previous_target = None
    for step, (temp, occ) in enumerate(PERCEPTS, start=1):
        target = reflex_decide(temp, occ)
        commands += 1                       # a reflex agent always transmits
        repeat = "  (repeat)" if target == previous_target else ""
        print(f"{step:<6}{temp:>7.1f}  {str(occ):<10}{target:<12}yes{repeat}")
        previous_target = target

    print("-" * 66)
    print(f"Commands transmitted: {commands} of {len(PERCEPTS)} steps")


if __name__ == "__main__":
    main()
