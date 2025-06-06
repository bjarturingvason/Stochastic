import matplotlib.pyplot as plt
import numpy as np
#THe LCG
def LCG(X0, m, a, c, N):
    X = np.zeros(N)
    X[0] = X0
    for i in range(1,N):
        X[i] = (a*X[i-1]+c) % m
    return X

# 1a)
X0 = int(1)
m = int(2**32)
a = int(1664525)
c = int(1013904223)
N = int(10000)

X = LCG(X0, m, a, c, N)
plt.hist(X,bins=10, rwidth=0.6)
plt.show()
plt.savefig('Figures/Hist-1a.png')

#1b)
fig, ax = plt.subplots()
ax.scatter(X[:-1], X[1:], s= 3)        # (u_n , u_{n+1})
ax.set_title("Successive pairs (uₙ , uₙ₊₁)")
ax.set_xlabel("uₙ")
ax.set_ylabel("uₙ₊₁")
plt.show()
plt.savefig('Figures/Pairscatter-1b')
