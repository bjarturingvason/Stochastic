import matplotlib.pyplot as plt
import numpy as np

X0 = int(1)
m = int(10000)
a = int(8)
c = int(17)

X = np.zeros(10000)
X[0] = X0
for i in range(1,10000):
  X[i] = (a*X[i-1]+c) % m

print((X))
plt.hist(X,bins=10, rwidth=0.6)
plt.show()