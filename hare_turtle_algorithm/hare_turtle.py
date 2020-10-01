"""
    This file contains implementation of the hare and turtle algorithm, it's
    for detecting cycles in a graph.

    * Let's use it to detect cycles in psuedo random generator

"""

def detect_cycle(f: callable, x0):
    # Phase I: find cycle
    T, H = f(x0), f(f(x0))
    while T != H:
        T, H = f(T), f(f(H))

    # Phase II: find position of the cycle
    T = x0
    while T != H:
        T = f(T)
        H = f(H)
    CycleStart = T

    # Phase III: length of the cycle
    Len = 1
    H = f(T)
    while T != H:
        H = f(H)
        Len += 1
    return Len, CycleStart

def main():
    f = lambda x: (888*x + 1)%2**16
    Len, CycleStart = detect_cycle(f, 0)
    print(f"Position of cycle start: {CycleStart}")
    print(f"cycle len: {Len}")
    print("Here is gonna be the cycle: ")

    # for _ in range(CycleStart, CycleStart + Len*3):
    #     print(CycleStart)
    #     CycleStart = f(CycleStart)



if __name__ == "__main__":
    main()