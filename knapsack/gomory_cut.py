"""
    Gomory cut is used for integer programming, and it's kind embedded
    into the Mixed Integer Programming algorithm.

    In this package, we are going to lower the level a bit and try to build the cut ourselves.

    * We are going to the Dual Simplex in Scipy instead of Pulp, which is not a native implementation.
"""

