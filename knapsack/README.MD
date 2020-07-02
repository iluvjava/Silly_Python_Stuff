## Dynamic Programming Knapsack ##

### Inputs ###
* Weights are positive.
* profits are positive. 
* All single item has a weight that is less than the maximum weight allowed. 

### Primal ALGO ###
* T[j, w] := the profits of items using sub array from 0 to j such that it sums up to weight of exactly w. 

* The recurrence relation: 
    * `T[j + 1, w] := max(T[j, w - weight[j]] + profits[j + 1], T[j, w])`
    * if `w - weight[j] < 0` then just ignore that case. 
    
* Base cases: 
    * `T[0, 0] := 0`
    * `T[0, w] := profit[0] if weight[0] == w else -inf`
    
* Optimization: 
    * only the previous column of T is used for each iteration, so that can be simplied by storing only previous column. 
    * We need to keep track of the solution too.

* Interpreting results: 
    * take max of `T[j, w]` for all possible w (sum of all weights)

### Dual Algo ###
* The above algorithm is the primal formulation of the problem, and there is a dual formulation of the same problem which can be solved with dynamic programming. 
* The alternative gives another kind of upper bound for the algorithm.
* Definition: 
    * `T[j, p]:` the minimum amount of weights such that it gives exactly a profits of c.  
* Recurrences: 
    * `T[j, p] := min(T[j - 1, p - p_j] + w_j, T[j - 1, p])`
    
### Motivation for Approximation algorithm ### 
* When both the weights and profits are real numbers, we need to use the Dual to approximate the solution with an interval. 

 

### Approximation Algorithm for Both Formulations ###

* Inputs: in addition to the DP algorithm a new parameter `epsilon` in (0, 1) is added. 


#### Primal's Approx ALGO ####
* Scale all numbers in the weights constraint using the multiplier: `log(w_max)/w_max` where w_max is the item with the maximum weights. 
    * This limits the size of the maximum weight, to a multiple of the numbers of items. 
    * Scale the Maximum weight allowed by the same factor, keep the problem unchanged. 
    * Scaling to a multiple of n is purely for mathematical beauty. 

* Round up the weights for each items, this makes all the weights an over estimation, this will provide an lower bound on profits of the solution. Round down the weights will provide an upper bound for the solution. 

* Compute the solution based on the rounded... weights... 
* ! This algorithm is arbitrarily bad, (Consider weights of 1, 1/2, 1/4, 1/8...), if it's off by a tiny round up decimal, then the profits can drop by 50% in this case. 
* Complexity: `O(N log(w_max))`, it's pretty efficient. 

#### Dual's Approx ALGO ####
* However, the dual problem gives an algorithm which produces precise lower bound, upper bound too. 
* Scale the profits of the items a way such that `p_max == n/epsilon` is true. 
* Scaled profits: `scale = n/(epsilon*p_max)`
* Then run dynamic programming algorithm on the scaled profits. 
* Then it's mathematically proven that the solution found, say `P_tilda`, is going to be `P_tilda >= (1-epsilon)*P_star`, where `P_star` is the optimal value on the solution with unscaled profits. 
* The complexity is `O(n^3/epislon)`. `O(n*p_sum)`, and the `p_sum <=p_max*n= n^2/epsilon`, then `O(n^3/epislon)` is the complexity of the algorithm. 
* If, n/epsilon is approximately the absolute optimal value, then the complexity becomes quadratic. 
* ! The upper bound is not the sum of exact profits on all the items in the solution, is `P_tailda/(1-epsilon)`, where the value `P_tilda` is the sum of rounded profits on the approximated solution. 

#### Integer Programming Approx ####
* Solving the problem as a linear programming problem will give an aboslute upper bound for all integers solutions for a given problem. This is the losest we can get, and the solution will be a set of fractions amount of items (exactly one of the item will have a fraction for it)
* Rank all items by `profits/weight`, choose until the last item that doesn't fit, and then we take fractional amount of the item to keep the solution feasible, which is the last item we can choose. 


## Branch and Bound and Lower/Upper Bound Bound Heuristics ##
![](img.png)
* Tuple `(I_f, I_0, I_1)` where each item represents: 
    * `I_1`: The solution we are definitely keeping. 
    * `I_0`: ~~~ we are not keeping. 
    * `I_f`: items we are still deciding. 
* Initialize `S*` as a lower bound for the real solution.
* (5) `S_tilda`: This is the solution where items in `I_free` is approximated (upper bound), and items in `I_1` is computed exactly.
* (6) Current choice of I_free and I_1 has an upper bound lower than global optimal, reject all I_free. 
* (7) `S_tilda` improves upper bound and feasible, mark that as current optimal. 
* (8) branch and bound, by neglecting/considering any random element from the `I_free` set. 

### Run-Time ###
* Exponential for Pathological Inputs. 

### Branch & Bound Approx 2 ###
* S* = Fraction_Approx(c, w, b) // approximate the solution with linear programming. 
* Define a recursive function: 
* `soln(c, w, b)` that solves the problem. 
```
def soln(S*, c, w, b): 
  upperBound:= "the upper bound of problem on c, w, b."
  S~ := "The Fractional solution of problem on c, w, b. "
  if upperBound <= profits(S*)
    return S*
  if S~ Integral 
    return S~ if profits(S~) > profits(S*) else return S*
  choose index 'i' of the item that has a fractional value from the approximation: 
    remove 'i' from c, w, keep the profits unchanged and solve this sub-problem. 
    keep 'i', but choose it as a whole, make b:= b - weight(i), and solve this sub problem. 
    return which ever has the higher profits. 
```

* The upper bound heuristic can be the minimum of all 3 heuristic algorithm. 


 