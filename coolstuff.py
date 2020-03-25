from typing import *
__all__ = ["Array"]


class Array():
    """
        It's a Multi dimensional array based on dictionary data structure.
        and it will stored stuff in a dictionary,
        with tuple matching with the values.

        Features:
            * Easy to instantiate
            * Memory efficient for huge array.
            * slicing; like matlab, or nparray.

        not a feature:
            * Indexing with negative integers.
    """
    def __init__(self, Indexranges: Tuple[int], defaultvalue = None):
        """

        :param Indexranges:
            It's a tuple, the length of the tuple is the dimension of the array
            the integer value in side the tuple is the range for that specific dimension.
        :param defaultvalue:
            If the index is not yet set, what is the default value you want to get from it?
        """
        for I in Indexranges:
            assert I > 0, "range of a certain dimension cannot be negative nor zero."
        self.__DefaultValue = defaultvalue
        self.__Dimension = len(Indexranges)
        self.__IndexRange = Indexranges
        self.__Map = {}
        return

    def size(self):
        return self.__IndexRange

    def get_specific_element(self, indices: Tuple[int]):
        """
            Receives a tuple with length matching the dimension of the array.
        :return:
            that specific element at that index of the array.
        """
        assert len(indices) == self.__Dimension, "Dimension mismatch"
        return self.__DefaultValue if indices not in self.__Map else self.__Map[indices]

    def set_specific_element(self, indices: Tuple[int], value):
        """
            Given a tuple representing a multi-dimensional index, it will return
            the value at the specific index.
        :param indices:
            A tuple
        :param value:
            The value,
        :return:
            the value, if not set before, the default value will be returned.
        """
        for I, J in zip(self.__IndexRange, indices):
            assert I > J >= 0, "Index out of range. "
        self.__Map[indices] = value
        return

    def slice(self, indices: Tuple[int]):
        """
            Slice a sub array inside the array.
            example:
                Let's say that the array "Arr" is 2d, m by n, then
                indices: (-1, 3)
                then it will slice (?, 3) and put then into an array with dimension (Arr.size()[0], 1)
                which is basically the third column, but as a mx1 array.
        :param indices:
            A list of tuple with dimension less than or equal to the dimension of the
            array.
        :return:
            Another instance of the Array class.
        """
        # check if the slicing index is valid:
        for I, E in enumerate(indices):
            assert -1<=E<=self.__IndexRange[I], "Invalid index for slicing."

        # Flatten the dimension from the slicing index:'
        NewDimension = []
        for I, E in enumerate(indices):
            NewDimension.append(self.__IndexRange[I] if E == -1 else 1)

        # Copying to a new multi-dimensional Array.
        NewArr = Array(tuple(NewDimension))

        def should_transfer(slicing, indices):
            for E1, E2 in zip(slicing, indices):
                pass


        pass

    def __getitem__(self, item:Tuple[int]):
        if len(item) == self.__Dimension:
            return self.get_specific_element(item)
        # Client want a sub array.
        raise Exception("Not Implemented yet")

    def __setitem__(self, key:Tuple[int], value):
        if len(key) == self.__Dimension:
            self.set_specific_element(key, value)
            return
        raise Exception("Not Implemented yet")

    def empty_array(self, *integers):
        return Array(integers)



def brief_test1():
    print("Setting and getting specif elements")
    arr = Array((3, 3, 3))
    assert arr[2, 2, 2] is None
    arr[1, 1, 1] = 9
    assert arr[1, 1, 1] == 9
    assert arr.size() == (3, 3, 3)
    print("brief_test1 END")



def main():
    print("main method run")
    brief_test1()
    pass


if __name__ == "__main__":
    main()