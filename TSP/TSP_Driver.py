from TSP.TSP_Core import TravelingSalesmanLP, rand_points
import matplotlib.pyplot as plt

def main():
    while True:
        RandPoints = rand_points([0, 10], [10, 0], 12)

        plt.scatter([T.x for T in RandPoints], [T.y for T in RandPoints])

        print(RandPoints)
        tlp = TravelingSalesmanLP()
        print(tlp)

        for p in RandPoints:
            tlp += p

        TspProblem = tlp.get_lp(10)
        TspProblem.solve()
        print(TspProblem.status)

        edges = []
        for i, e in tlp.e.items():
            if e.varValue == 1:
                if i[0] != i[1]:
                    print(i, tlp.points[i])
                    edges.append(i)

        for e in edges:
            V = tlp.points
            p1, p2 = V[e[0]], V[e[1]]
            plt.plot([p1.x, p2.x], [p1.y, p2.y])
        print(edges)
        plt.show()

        if input() == "q":
            break




if __name__ == "__main__":
    main()