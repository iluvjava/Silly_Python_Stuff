"""
    An ordered muti-set
    * This is created with the purpose of solving some nasty hackerrank problem.
"""

from collections import Counter


class OrderedMultiSet:

    """
        Created with the purpose of storing the symbols set for the sequence for the problem.
        * This is going to be an iterator.
        * This is going to support subset operation.
        * Supports removal of one element at a time while iterating through, but it has to
        reset after one complete iteration.
        Author: Alto
    """

    def __init__(self, allSymbols):
        """
            Construct the multi set from a list of symbols
        """
        from collections import Counter
        MultiSet = Counter(allSymbols)
        SortedUniqueSymbols = sorted(list(set(allSymbols)))
        self._Cardinality = len(allSymbols)
        self._MtSet = MultiSet
        self._Symbols = SortedUniqueSymbols
        self._Pointer = (0, 0)  # Pointing the the element that is going to be returned in the multiset.
        self._Toreturn = None  # A back for the element to return, so that, we don't have array out of index when the
        # element is removed immediately after the use of .next().
        self._PrePointer = None  # Pointing at the element that is going to be returned by the next call of next()
        """
            self._Pointer: 
              This is the future pointer, it might, or might not point to a valid value, this is true because, 
              after the remove() has been called, it might failed to point to anything, that is where, self._Toreturn
              comes in. 
            
            self._Toreturn:
              This is a back up, when next() is called, it prepare the next value that is is going to be returned 
              next time in advance, to prevent self._Pointer to be pointing into invalid places, and self._Pointer
              will be reset to a valid pointer once the function next() is done. 
              
            self._PrePointer: 
              When removing the element that was returned by next(), we need to know which element it is, and that is 
              what this variable is for.
              Just goes to the this pointer in the multi-set and remove the element, or decremenet the frequency of the 
              element accordingly. 
        """

    def reset_itr(self):
        self._Pointer = (0, 0)
        self._Toreturn = None
        self._PrePointer = None

    def remove(self):
        """
            Remove the most recent element that is returned by the next() method.
            * The pointer is pointing at the future element, not the most recent element
        """
        assert self._PrePointer is not None, "Can't remove when you haven't called next()"
        ToRemoveIdx, _ = self._PrePointer
        ToRemoveElement = self._Symbols[ToRemoveIdx]
        if self._MtSet[ToRemoveElement] == 1:
            del self._MtSet[self._Symbols[ToRemoveIdx]]
            self._Symbols.pop(ToRemoveIdx)
        else:
            self._MtSet[ToRemoveElement] -= 1

        self._Cardinality -= 1

    def next(self):
        """
            Returns the current element from from the multiset.
            * Return current element, move the pointer to the next element.
        """
        assert self.has_next(), "can't get next when it's the last one"
        P1, P2 = self._Pointer
        self._PrePointer = self._Pointer
        Results = self._Symbols[P1]
        if P2 >= self._MtSet[P1]:
            self._Pointer = (P1 + 1, 0)
        else:
            self._Pointer = (P1, P2 + 1)
        # Pointer has been advanced
        if self._Pointer[0] <= len(self._Symbols) - 1:
            self._Toreturn = self._Symbols[self._Pointer[0]]
        else:
            self._Toreturn = None  # no more elements folks, I am sorry.
        return Results

    def has_next(self):
        P1, F1 = self._Pointer
        if P1 >= len(self._Symbols):
            return False
        if self._MtSet[P1] > F1:
            return P1 < len(self._Symbols) - 1
        if P1 >= len(self._Symbols) - 1:
            return F1 < self._MtSet[self._Symbols[P1]]

        return True

    def subset_of(self, other):
        for K, V in self._MtSet.items():
            if K not in other or other[K] < self._MtSet[K]:
                return False
        return True

    @property
    def Cardinality(self):
        return self._Cardinality

    def __repr__(self):
        return str(dict(self._MtSet))


def main():
    def TestCase1():
        Sequence = list([C for C in "abcdabdc"])
        Mset = OrderedMultiSet(Sequence)
        print(Mset)
        print("Removing all letter: 'a' while iterator throug")
        while Mset.has_next():
            Token = Mset.next()
            print(f"Looking at: {Token}")
            if Token == "a":
                Mset.remove()
        assert Mset.subset_of(Counter("bcbcdd"))
        assert not Mset.subset_of(Counter("abcabc"))
        assert Mset.Cardinality == 6
        print(Mset)

    def TestCase2():
        from random import shuffle
        Charlist = [C for C in "abdefg"*14] + ["c"]*20
        shuffle(Charlist)
        Mset = OrderedMultiSet(Charlist)
        print(Mset)
        print("Removing all letter: 'a' while iterator throug")
        while Mset.has_next():
            Token = Mset.next()
            print(f"Looking at: {Token}")
            if Token == "c":
                print("that one got removed")
                Mset.remove()
        print(Mset)

    def Testcase3():
        Charlist = [C for C in "abcdefgghhii"]
        Mset = OrderedMultiSet(Charlist)



    TestCase1()
    TestCase2()


if __name__ == "__main__":
    main()