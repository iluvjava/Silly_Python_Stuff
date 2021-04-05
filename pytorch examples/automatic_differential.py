import torch

tensor = torch.Tensor
mm = torch.mm
jacobian = torch.autograd.function.jacobian

def BackwardsAutoDiff():
    """
        Using automatic backwards differential to figure out the
        differential of a scalar function.
    :return:
    """

    def exampleFunc(x, A):
        xT = x.transpose(0, 1)
        mm(xT, mm(A, x))


    pass

def BasicTensorsManipulation():
    x = tensor([1, 2, 3, 4])
    y = tensor([2, 3, 4, 5])
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)
    print(x**y)



def main():
    BasicTensorsManipulation()

if __name__ == "__main__":
    import os
    print(f"dir: {os.curdir}")
    print(f"wd: {os.getcwd()}")
    main()