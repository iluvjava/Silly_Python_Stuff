import json
import os
CWD = os.getcwd() # Current working directory.
print(f"CWD: {CWD}")
import matplotlib.pyplot as pyplt
PYPLOTCOLORS = ['b', 'g','r', 'c','m', 'y', 'k']


def read_str_from(fileName:str):
    with open(fileName, "r") as OpenedFile:
        return "".join(OpenedFile.readlines())


def plot_nested_point_list(pointListList):
    for I, PointList in enumerate(pointListList):
        if I > len(PYPLOTCOLORS):
            break
        for Point in PointList:
            x, y = Point["x"], Point["y"]
            pyplt.scatter(x=x, y=y, color=PYPLOTCOLORS[I])
    pyplt.show(dpi="400")


def main():

    for I in range(1, 9):
        PointListList = json.loads(read_str_from(f"test2_clusters{I}.json"))
        plot_nested_point_list(PointListList)
    print(PointListList)



if __name__ == "__main__":
    main()