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
        return 0 <= index and index < len(arr)

    def filterSingleRepeatition(indices):
        newindices = [indices[0]]
        for I in range(1, len(indices)):
            if indices[I] - indices[I - 1] == 1:
                continue
            newindices.append(indices[I])
        return indices

    III = II.copy()
    III = filterSingleRepeatition(III)
    AggregateSymbols = {} # Wiggle loci for each index... maybe later.

    # Assume all indices in III is pointing the the exact same symbol, register visited indices:
    for I in III:
        AggregateSymbols[I] = S[I]
        VistiedIndicesInS.add(I)

    # right wiggle
    def wiggle(d):
        TotalWiggle = 0
        while(len(III) != 0):

            # If collided OR out of range, no right wiggle.
            temp = III.copy()
            for I in temp:
                if (I + d in VistiedIndicesInS) or (not IndexInRange(S, I + d)):
                    III.remove(I)
            del temp

            # check right symbol, if unique, then no right wiggle.
            AdjSymbols = {}
            for I in III:
                if S[I + d] not in AdjSymbols:
                    AdjSymbols[S[I + d]] = [I]
                else:
                    AdjSymbols[S[I + d]].append(I)

            for K in AdjSymbols.keys():
                if len(AdjSymbols[K]) == 1:
                    III.remove(AdjSymbols[K][0])

            # right wiggle and mark visited.
            for I,V in enumerate(III):
                VistiedIndicesInS.add(V + d)
                AggregateSymbols[V - TotalWiggle] = AggregateSymbols[V - TotalWiggle] + S[V + d] if d == 1\
                    else S[V + d] + AggregateSymbols[V - TotalWiggle]
                III[I] += d

            TotalWiggle += d

    wiggle(1)
    III = II.copy()
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

    print(GetRepeatedSubSequenceContaining("111212123123098765", '1'))
    print(GetRepeatedSubSequenceContaining("111112222yuiyuiyuiopyuiyuiyuiopll", '1'))
    print(GetRepeatedSubSequenceContaining("111112222yuiyuiyuiopyuiyuiyuiopll", 'y'))
    print(GetRepeatedSubSequenceContaining("123123", '1'))
    print(GetRepeatedSubSequenceContaining("1212123123123412341234512345123456123456", '1'))
    print(GetRepeatedSubSequenceContaining("12121212", '1'))
    print(GetRepeatedSubSequenceContaining("1212123", '1'))
    print(GetRepeatedSubSequenceContaining("123 123123 123", '1')) # Nested repeated pattern
    print(GetRepeatedSubSequenceContaining("123321123321", '1'))  # Nested repeated pattern

    print(GetRepeatedSubSequenceContaining("12121", '1'))  # Intersected Repeating Sequences

    # Inverse
    print(GetRepeatedSubSequenceContaining("321321", '1'))
    print(GetRepeatedSubSequenceContaining("43214321", '1'))

    print(GetRepeatedSubSequenceContaining("4321043210", '1'))

    print(GetRepeatedSubSequenceContaining("78901234567890123456", '1'))

if __name__ == "__main__":
    main()