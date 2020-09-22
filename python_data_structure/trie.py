"""



"""
class LetterTrie:

    """
        This is a data structure ths design to improve runtime of some of the coding challenges.

        * This represent a set of sequences containing 26 letters in the lower case with the space character.
        * Give the prefix of the sequences, it returns all the sequences in the set such that they all start
        with that prefix.

        * This class is going to be implemented with nested array, with None representing the termminal
        condition on one of the branches.
            * So this is also gonna be sorted too.


    """
    def __init__(self):
        self.__FirstLayer = [None]*27

    def __to_index(self, letter):
        if letter == " ":
            return 27
        return ord(letter) - ord("a")


    def __to_letter(self, indx):
        if indx == 27:
            return " "
        return chr(indx + ord("a"))

    def add(self, letters):
        LetterList = list(letters)
        Layer = self.__FirstLayer
        for L in LetterList:
            L = self.__to_index(L)
            if Layer[L] is None:
                Layer[L] = [None]*27
            Layer = Layer[L]


    def __traverse(self, layer, prefix, accumulator=None):
        accumulator = [] if accumulator is None else accumulator
        if layer is None:
            accumulator.append(prefix)
            return
        for I, E in enumerate(layer):
            if E is not None:
                Letter = self.__to_letter(I)
                self.__traverse(E, prefix + Letter, accumulator)

        return accumulator

    def get(self, prefix):
        # Get to the correct layer
        Layer = self.__FirstLayer
        for L in prefix:
            Idx = self.__to_index(L)
            Layer = Layer[Idx]
        return self.__traverse(Layer, prefix)

    def __repr__(self):
        return str(self.__traverse(self.__FirstLayer, ""))


class Trie():
    def __init__ (self):
        self.__Layer = {}  # this is the first layer

    def add(self, sequence):
        sequence = list(sequence)
        Layer = self.__Layer
        for L in sequence:
            if L not in Layer:
                Layer[L] = {}
            Layer = Layer[L]


    def __traverse(self, startingLayer, prefix="", accumulate=None):
        accumulate = [] if accumulate is None else accumulate
        # if startingLayer is None:
        #     # Prefix is not even in the first layer.
        #     return accumulate
        for Letter, NextLayer in startingLayer.items():
            if NextLayer:
                self.__traverse(NextLayer, prefix + Letter, accumulate)
            else:
                if NextLayer is None:
                    pass
                else:
                    accumulate.append(prefix + Letter)
        return accumulate

    def __layer_traverse(self, prefix):
        Layer = self.__Layer
        for Letter in prefix:
            if Letter not in Layer:
                return None
            Layer = Layer[Letter]
        return Layer

    def delete(self, sequence):
        Layer = self.__layer_traverse(sequence[:-1])
        Layer[sequence[-1]] = None
        pass

    def get(self, prefix):
        if prefix[0] not in self.__Layer:
            return []
        return self.__traverse(self.__layer_traverse(prefix), prefix=prefix)


    def __repr__(self):
        return str(self.__traverse(self.__Layer))


def main():
    T = Trie()
    T.add("bbc")
    T.add("bbd")
    T.add("aac")
    T.add("bcd")
    print(T)
    print(T.get("d"))
    print(T.get("a"))
    print(T.get("b"))
    print(T.get("bb"))
    T.delete("bbc")
    print(T.get("bb"))
    pass


if __name__ == "__main__":
    main()


