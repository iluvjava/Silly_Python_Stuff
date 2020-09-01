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
        pass

    def peek(self):
        pass

    def remove(self, element):
        pass

    def __percolate_up(self, idx):
        """
            Does it recursively
        :param idx:
        :return:
        """
        H, P = self._Heap, self.__get_parent
        D = self._Index
        if H[idx] < H[P[idx]]:  # swap with parent.
            D[H[idx]], D[H[P[idx]]] = D[H[P[idx]]], D[H[idx]]
            H[idx], H[P[idx]] = H[P[idx]], H[idx]
        else:  # Base case
            return idx
        return self.__percolate_up(P[idx])

    def __percolate_down(self, idx):
        H, P = self._Heap, self.__get_parent
        D = self._Index

        pass

    def __percolate(self, idx):
        return self.__percolate_up(self.__percolate_up(idx))

    def __get_children(self, idx):
        return 2*idx + 1, 2*idx + 2

    def __get_parent(self, idx):
        return round(idx/2) - 1



def main():
    pass

if __name__ == "__main__":
    main()