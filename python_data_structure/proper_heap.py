"""
    So we are going to make the heap that exists in python a bit better.

    * Supports for removal of elements from the heap.
    * Supports for repeating elements in the heap.
    * It's a binary heap.

    Some Notes for the programmer:

    Method: push:
      * Use the heaq push if the element doesn't exist, update it to the index/freq trackers.

    Method: peek:
      * Get the reference to the first element in the array.

    MNethod: remove:
      * locate the element, and decrease the frequencies. If the decrementation of the frequrency goes to 0
        * Pop last element from the array, replace it to the element wanting to be removed.
        * Then percolate up/down the heap on the new element.

    Some Warning from the programmer:
        * If A <= B and B <= A, then A==B has to be true, the rules of order field must be applicable to the
        data type you are using with this heap class.

"""

class Heap:
    """
        We are going to extend the heapq in python
    """

    def __init__(self):
        self._Heap = []  # The heap storing unique elements in the heap
        self._Freq = {}  # A mapping from the element to the array to the repeatition of the that elements in the heap.
        self._Index = {}  # The reverse mapping from elements in the array to the index of that element.
        self._Count = 0

    def push(self, element):
        self._Count += 1
        if element in self._Freq:
            self._Freq[element] += 1
            return
        self._Heap.append(element)
        self._Freq[element] = 1
        self._Index[element] = len(self._Heap) - 1
        self.__percolate_up(len(self._Heap) - 1)
        return self.__heap_invariant_assert()

    @property
    def size(self):
        """
        The number of unique elements in the heap.
        :return:
        """
        return len(self._Heap)

    @property
    def count(self):
        """
            All the elements counting the repetitions
        :return:
        """
        return self._Count

    def peek(self):
        return self._Heap[0]

    def pop(self):
        ToReturn = self._H[0]
        self.remove(self._H[0])
        return ToReturn

    def remove(self, element):
        # Frequencies decrementations ----------------------------------------------------------------------------------
        F = self._Freq
        H = self._Heap
        D = self._Index
        if element not in F:
            raise RuntimeError("Cannot remove an element that is not in the heap")
        if F[element] == 1:
            del F[element]
            ToRemoveIndex = D[element]  # Idx of the element to remove
            ReplacementIndex = self.size - 1
            D[H[ReplacementIndex]] = ToRemoveIndex
            H[ToRemoveIndex] = H[ReplacementIndex]  # overwrite the element
            H.pop()
            del D[element]
            if ToRemoveIndex != ReplacementIndex:
                self.__percolate(ToRemoveIndex)
        else:
            F[element] -= 1
        self._Count -= 1
        return self.__heap_invariant_assert()

    def __percolate_up(self, idx):
        """
            Does it recursively
        :param idx:

        :return:
        """
        if idx == 0:  # Can't percolate up anymore.
            return 0
        H, P = self._Heap, self.__get_parent(idx)
        D = self._Index
        if H[idx] < H[P]:  # swap with parent.
            D[H[idx]], D[H[P]] = D[H[P]], D[H[idx]]
            H[idx], H[P] = H[P], H[idx]
        else:  # Base case
            return idx
        return self.__percolate_up(P)

    def __percolate_down(self, idx):
        """
            Percolate down is percolate its children up if it has any.
        :param idx:
        :return:
        """
        C1, C2 = self.__get_children(idx)
        H = self._Heap
        D = self._Index
        C = None
        if not(C1 is None or C2 is None):
            if H[C1] < H[C2]:
                C = C1
            else:
                C = C2
        else:
            C = C1 if C1 is not None else C2
        if C is None:  # no children to swap down
            return idx
        if H[idx] > H[C]:
            D[H[C]], D[H[idx]] = D[H[idx]], D[H[C]]
            H[C], H[idx] = H[idx], H[C]
            return self.__percolate_down(C)
        else:
            return idx

    def __percolate(self, idx):
        return self.__percolate_down(self.__percolate_up(idx))

    def __get_children(self, idx):
        """

        :param idx:
        :return:
            None if that children is out of the heap
        """
        C1, C2 = 2*idx + 1, 2*idx + 2
        C1 = C1 if C1 < len(self._Heap) else None
        C2 = C2 if C2 < len(self._Heap) else None
        return C1, C2

    def __get_parent(self, idx):
        import math as m
        return m.ceil(idx/2) - 1

    def __repr__(self):
        s = str(self._Heap) + "\n" + str(self._Freq) + "\n"
        s += f"CountUnique: {self.size}\n"
        s += f"CountAll: {self._Count}\n"
        return s

    def __heap_invariant_assert(self):
        """
            This is an internal debug method for the data structure.
            TODO: Delete after testing.
        :return:

        """
        for I in range(self.size):
            P = self._Heap[I]
            C1, C2 = self.__get_children(I)
            if C1 is not None:
                assert self._Heap[C1] > P, f"Failed on index: {I}, object state: {self}"
            if C2 is not None:
                assert self._Heap[C2] > P, f"Failed on index: {I}, object state: {self}"
        return

def main():

    def Test1_JustPushing():
        TheHeap = Heap()
        for I in reversed(range(8)):
            print(f"adding: {I}")
            TheHeap.push(I)
        print(TheHeap)
        return TheHeap

    def Test2_Justremoving():
        TheHeap = Test1_JustPushing()
        from random import shuffle as shuff
        RandomList = list(range(TheHeap.size))
        shuff(RandomList)
        for I in RandomList:
            print(f"Removing element: {I}")
            TheHeap.remove(I)
            pass
        pass

    def Comprehensive_Testing():
        # Use top k_sort for it.
        from random import random
        RandomList = [int(random()*10) for _ in range(100)]
        SortedList = sorted(RandomList)  # for refernce.
        TheHeap = Heap()
        for E in RandomList:
            print(f"Pushing: {E}")
            TheHeap.push(E)
        print(TheHeap)


    Test1_JustPushing()
    Test2_Justremoving()
    Comprehensive_Testing()


if __name__ == "__main__":
    main()