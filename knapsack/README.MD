## Dynamic Programming Knapsack ##

### inputs ###
* Weights are positive.
* profits are positive. 

### ALGO ###
* T[j, w] := the profits of items using sub array from 0 to j such that it sums up to weight of exactly w. 

* The recurrence relation: 
    * `T[j + 1, w] := min(T[j, w - weight[j]], T[j, w])`
    
* Base cases: 
    * `T[0, 0] := 0`
    * `T[0, w] := profit[0] if weight[0] == w else -inf`
    
* Optimization: 
    * only the previous column of T is used for each iteration, so that can be simplied by storing only previous column. 
    * We need to keep track of the solution too.

* Interpreting results: 
    * take max of `T[j, w]` for all possible w (sum of all weights)


 
 