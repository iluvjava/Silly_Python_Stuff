class KahanRunningSumMutable:

    def __init__(self, initialSum):
        self.__Sum = initialSum
        self.__Compensator = 0

    @property
    def Sum(self):
        return self.__Sum + self.__Compensator  # must round it. Because the final difference might make

    def add(self, other):
        """
                   Add a number to the sum.
               :param other:
                   Float, ints, or whatever.
               :return:
               """
        Temp = self.__Sum + other
        if abs(self.__Sum) >= abs(other):
            self.__Compensator += (self.__Sum - Temp) + other
        else:
            self.__Compensator += (other - Temp) + self.__Sum
        self.__Sum = Temp
        return self

    def multiply(self, other):
        self.__Sum = round(other*self.__Sum, 16)
        self.__Compensator = round(other*self.__Compensator, 16)
        return self
