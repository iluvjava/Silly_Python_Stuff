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
"""



class Heap:
    """
        We are going to extend the heapq in python.

    """
    def __init__(self):
        self._Heap = []  # The heap storing unique elements in the heap
        self._Freq = {}  # A mapping from the element to the array to the repeatition of the that elements in the heap.
        self._Index = {}  # The reverse mapping from elements in the array to the index of that element.

    def push(self, element):
        if element in self._Freq:
            self._Freq[element] += 1
            return
        self._Heap.append(element)
        self._Freq[element] = 1
        self._Index[element] = len(self._Heap) - 1
        self.__percolate_up(len(self._Heap) - 1)
        return

    def peek(self):
        return self._Heap[0]

    def remove(self, element):

        pass

    def __percolate_up(self, idx):
        """
            Does it recursively
        :param idx:

        :return:
        """
        H, P = self._Heap, self.__get_parent(idx)
        D = self._Index
        if H[idx] < H[P]:  # swap with parent.
            D[H[idx]], D[H[P]] = D[H[P]], D[H[idx]]
            H[idx], H[P] = H[P], H[idx]
        else:  # Base case
            return idx
        return self.__percolate_up(P(idx))

    def __percolate_down(self, idx):
        """
            Percolate down is percolate its children up if it has any.
        :param idx:
        :return:
        """
        C1, C2 = self.__get_children(idx)
        H = self._Heap
        D = self._Index
        C = C1 if C1 is not None else None
        C = C if C is not None else C2
        if C is None:  # no children to swap down
            return idx
        if H[idx] > H[C]:
            D[C], D[idx] = D[idx], D[C]
            H[C], H[idx] = H[idx], H[C]
            return self.__percolate_down(C)
        else:
            return idx

    def __percolate(self, idx):
        return self.__percolate_up(self.__percolate_up(idx))

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
        return round(idx/2) - 1



def main():
    pass

if __name__ == "__main__":
    main()