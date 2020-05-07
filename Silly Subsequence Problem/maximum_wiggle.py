"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Problem Statement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
S := a sequence of symbols
I := a set of valid indices for S.

    * 2 index wiggle:
    let i, j in I,
    They can wiggle 'n' unit to the right if the following is true:
        i + n < j
        for all k in range(n):
            S[i+k] == s[j + k]

    * m index wiggle:
    let II be a subset of I,
    for each i in II, it can right wiggle k if:
        at least one of i in II satisfies:
            i + k < II[II.indexof(i) + 1]
        AND
        at least 2 of i,j in II satisfies:
            for q in range(K):
                S[i + q] = s[j + q]

    left wiggle and right wiggle is same but in reverse.

"""

def TryWiggle(II, S):
    if len(II) <= 1:
        return None
    VistiedIndicesInS = set()

    def IndexInRange(arr, index):
        return 0 <= index < len(arr)
    def filterSingleRepeatition(indices):
        newindices = [indices[0]]
        for I in range(1, len(indices)):
            if indices[I] - newindices[-1] == 1:
                continue
            newindices.append(indices[I])
        return newindices

    III = II.copy()
    III = filterSingleRepeatition(III)
    AggregateSymbols = {} # Wiggle loci for each index... maybe later.

    # Assume all indices in III is pointing the the exact same symbol, register visited indices:
    for I in III:
        AggregateSymbols[I] = S[I]
        VistiedIndicesInS.add(I)

    # right wiggle
    def wiggle(d):
        RightTotalWiggle = 0
        while(len(III) != 0):
            # If collided OR out of range, no right wiggle.
            for I in III:
                if I + d in VistiedIndicesInS or not IndexInRange(S, I + d):
                    III.remove(I)

            # check right symbol, if unique, then no right wiggle.
            RightSymbols = {}
            for I in III:
                if S[I + d] not in RightSymbols:
                    RightSymbols[S[I + d]] = [I]
                else:
                    RightSymbols[S[I + d]].append(I)

            for K in RightSymbols.keys():
                if len(RightSymbols[K]) == 1:
                    III.remove(RightSymbols[K][0])

            # right wiggle and mark visited.
            for I,V in enumerate(III):
                VistiedIndicesInS.add(V + d)
                AggregateSymbols[V - RightTotalWiggle] += S[V + d]
                III[I] += 1

            RightTotalWiggle += 1
    wiggle(1)
    wiggle(-1)

    return VistiedIndicesInS, AggregateSymbols

def GetRepeatedSubSequenceContaining(S, character: chr):
    _, res = TryWiggle([I for I, L in enumerate(S) if L == character], S)
    return res

def main():
    S = "144133133144100"
    Visted, SubSequences = TryWiggle([I for I, L in enumerate(S) if L == '1'], S)
    print(Visted, SubSequences)
    Visted, SubSequences = TryWiggle([I for I, L in enumerate(S) if L == '3'], S)
    print(Visted, SubSequences)

    print("Cool, let's try some new shit. ")

    pass


if __name__ == "__main__":
    main()