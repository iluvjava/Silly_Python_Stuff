class IntegerMedianKeeper():
    """
    Author: Alto Legato
        The range of all the integer data mustbe predefined.
        All must be positive integers.
    """

    def __init__(self, maxIntVal):
        self._N = 0
        self._P = (0, 0)  # index, frequency pointer, fixed at n//2 if even, n//2 + 1 if odd
        self._Range = (0, maxIntVal)
        self._Arr = [0 for _ in range(maxIntVal + 1)]

    def add(self, E):
        if E < self._Range[0] or E > self._Range[1]:
            raise RuntimeError
        I, J, Arr = self._P[0], self._P[1], self._Arr
        self._N += 1
        Arr[E] += 1
        # Edge case size == 1 ------------------------------------------------------------------------------------------
        if self._N == 1:
            self._P = (E, 1)
            return
        # Edge case size == 2
        if self._N == 2 and E < I:
            self.__move_left()
            return
        # Decide which direction to move the pointer -------------------------------------------------------------------
        if E >= I and self._N % 2 == 1:
            self.__move_right()
        elif E < I and self._N % 2 == 0:
            self.__move_left()


    def remove(self, E):
        if self._Arr[E] == 0:
            raise RuntimeError
        I, J, Arr = self._P[0], self._P[1], self._Arr
        self._Arr[E] -= 1
        # Decide where to move the pointer -----------------------------------------------------------------------------
        if E < I and self._N % 2 == 0:
            self.__move_right()
        elif E > I and self._N % 2 == 1:
            self.__move_left()
        elif E == I: # Edge case ---------------------------------------------------------------------------------------
            if J > Arr[I]:
                if self._N % 2 == 1:
                    self.__move_left()
                else:
                    self.__move_right()
            else:
                if self._N%2 == 1:
                    self.__move_left()
        self._N -= 1




    def __move_left(self):
        """
            Find the immediate number on the left of
            current index that is not empty.
        :param toRight:
        :return:
        """
        I, J = self._P
        Arr = self._Arr
        if J - 1 >= 1:
            self._P = (I, J - 1)
        else:
            I -= 1
            while Arr[I] == 0:
                I -= 1
            self._P = (I, Arr[I])


    def __move_right(self):
        I, J = self._P
        Arr = self._Arr

        if J + 1 <= Arr[I]:
            self._P = (I, J + 1)
        else:
            I += 1
            while Arr[I] == 0:
                I += 1
            self._P = (I, 1)

    def __repr__(self):
        S = "frequencies: \n"
        Freq = {}
        for I, V in enumerate(self._Arr):
            if V != 0:
                Freq[I] = V
        S += f"{Freq}; "
        S += f"Point pointing at the {self._P[1]} th element of {self._P[0]}; "
        S += f"N: {self._N}"
        return S


    def median(self):
        I, J = self._P
        if self._N == 1:
            return I
        Arr = self._Arr
        if self._N % 2 == 0:
            if Arr[I] == J:
                I += 1
                while Arr[I] == 0:
                    I += 1
            return (I + self._P[0]) / 2
        else:
            return self._P[0]


    @property
    def arr(self):
        return self._Arr


def sort_find_median(arr):
    if len(arr) == 1:
        return arr[0]
    arr = sorted(arr)
    I, J = arr[len(arr)//2 - 1], arr[len(arr)//2]
    return J if len(arr)%2 == 1 else (I + J)/2


def main():

    from random import random as rnd
    def Test2():
        N = 10000
        ReferenceArray = []
        Keeper = IntegerMedianKeeper(100)
        for I in [int(rnd()*10 + 1) for _ in range(N)]:
            ReferenceArray.append(I)
            Keeper.add(I)
            M1, M2 = sort_find_median(ReferenceArray) , Keeper.median()
            assert M1 == M2, f"{M1} != {M2}"
        return

    def Test3():
        N, trials = 10, 1000
        D = 10
        ReferenceArray = [int(rnd()*N) + 1 for _ in range(D)]
        Keeper = IntegerMedianKeeper(N)
        for E in ReferenceArray:
            Keeper.add(E)
        for R in [int(rnd()*N) + 1 for _ in range(trials)]:
            Keeper.add(R)
            ReferenceArray.append(R)
            Keeper.remove(ReferenceArray[0])
            ReferenceArray.pop(0)
            M1, M2 = sort_find_median(ReferenceArray), Keeper.median()
            assert M1 == M2, f"{M1} != {M2}, state of the object: \n{Keeper}"

    Test3()

if __name__ == "__main__":
    main()