
__all__ = ["UnionFind", "UnionSetJoin"]
"""
    Features:
        * Use the element added by the user as representatives of the groups.
        * Join(a, b) means the new representative of a, is the representatie of b, no abiguity.
          * Tree Rank machanics is gonna be broken.
        * Implemented using a map.

"""

class UnionFind:

    def __init__(self):
        self._M = {}
        pass

    def __call__(self, element):
        """
            get the representative of the element.

            * If the element is never added, then it will be added
            to the data structure and it will return itself.

            * Path compression
        :return:
            Representative or itself in the case that the element
            never really appeared in the data-structure before.
        """
        if element not in self._M:
            self._M[element] = None  # Root.
            return element

        if self._M[element] is None:
            return element
        Representative = self(self._M[element])
        self._M[element] = Representative
        return Representative

    def join(self, a, b):
        Represent1 = self(a)
        Represent2 = self(b)
        if Represent1 == Represent2:
            return
        self._M[Represent1] = Represent2


class UnionSetJoin(UnionFind):
    """
        The representative of the joined sets are now literally a
        set with all the elements that got joined together.
    """

    def __init__(self):
        super().__init__()
        self._ElementToInts = {}
        self._Sets = []

    def __call__(self, element):
        if element not in self._ElementToInts:
            self._ElementToInts[element] = len(self._Sets)
            self._Sets.append(set([element]))
            return super().__call__(element)
        return super().__call__(element)

    def join(self, a, b):
        if self(a) == self(b):
            return
        SetReprOfB = self._Sets[self._ElementToInts[self(b)]]
        SetReprOfB |= self._Sets[self._ElementToInts[self(a)]]
        super().join(a, b)

    def get_joined_set(self, element):
        return self._Sets[self._ElementToInts[self(element)]]

    def __repr__(self):
        s = ""
        for K in self._ElementToInts:
            s += f"{K}: {self.get_joined_set(K)}\n"
        return s


if __name__ == "__main__":
    def main():
        def Test1():
            Uf = UnionFind()
            for I in range(5):
                Uf(I)
            Uf.join(1, 2)
            Uf.join(2, 3)
            print(Uf(1))

        def Test2():
            Uf = UnionSetJoin()
            for I in range(5):
                Uf(I)
            Uf.join(1, 2)
            Uf.join(2, 3)
            print(Uf(1))
            print(Uf)

        Test1()
        print()
        Test2()

    main()