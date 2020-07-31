# def faray_sequence(N):
#     F0 = [(0, 1), (1,1)]
#     for _ in range(N):
#         FNext = []
#         for I in range(len(F0) - 1):
#             FNext.append(F0[I])
#             FNext.append((F0[I][0] + F0[I+1][0], F0[I][1] + F0[I+1][1]))
#         FNext.append(F0[-1])
#         F0 = FNext
#     return FNext

def farey_sequence(n: int, descending: bool = False) -> None:
    """
        Print the n'th Farey sequence. Allow for either ascending or descending.
        Copied from wikipedia.
    """
    (a, b, c, d) = (0, 1, 1, n)
    if descending:
        (a, c) = (1, n - 1)
    print("{0}/{1}".format(a,b))
    while (c <= n and not descending) or (a > 0 and descending):
        k = (n + b) // d
        (a, b, c, d) = (c, d, k * c - a, k * d - b)
        print("{0}/{1}".format(a,b))

def main():
    print(farey_sequence(5))


if __name__ == "__main__":
    main()