import numpy as np

y = x = np.array([np.arange(0, 10)]).T

print(x.T.shape)
print(y.shape)
print(x@y.T)

print(np.linspace(0, 100, 400)[np.newaxis, :].T)

