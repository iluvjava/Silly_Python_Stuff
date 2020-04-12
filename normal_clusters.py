import numpy as np
import matplotlib.pyplot as plt
import math as mth

def main():
    n = 300
    cluster1 = NormalSphericalDistribution3D((0,0,0), sigma=20, samplesize=n)
    cluster2 = NormalSphericalDistribution3D((60,60,60), sigma=20, samplesize=n)
    merged = cluster2 + cluster1

    print(f"len(merged) = {len(merged)}")
    edgesWeights = []
    for I in range(len(merged)):
        for J in range(I + 1, len(merged)):
            edgesWeights.append(dis(
                merged[I], merged[J]))

    edgesWeights = sorted(edgesWeights)

    with open("table.txt", 'w') as f:
        for Value in edgesWeights:
            f.write(f"{Value}\n")

    print("2 Cluster Closest Distance: ")
    print(min(dis(v1, v2) for v1 in cluster1 for v2 in cluster2))

    pass

def BreakAxis(data, parts):
    Min = mth.floor(min(data))
    Max = mth.ceil(max(data))
    binwidth = (Max - Min)//parts
    return range(Min, Max + binwidth, binwidth)

def NormalSphericalDistribution3D(center, sigma, samplesize):
    XcoordList = np.random.normal(center[0], sigma, samplesize)
    YcoordList = np.random.normal(center[1], sigma, samplesize)
    ZcoordList = np.random.normal(center[2], sigma, samplesize)
    points = [(x, y, z) for x, y, z in zip(XcoordList, YcoordList, ZcoordList)]
    return points

def dis(x, y):
    return sum(abs(a-b) for a,b in zip(x, y))


if __name__== "__main__":
    main()
