
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
from scipy import stats

#preventive-treatment matrix  Q (fill * with negatives such that the rows sum to 0)
Q_trt = np.array([
    [-0.00475, 0.0025 , 0.00125, 0.    , 0.001 ],
    [ 0.    , -0.007 , 0.     , 0.002 , 0.005 ],
    [ 0.    ,  0.    , -0.008 , 0.003 , 0.005 ],
    [ 0.    ,  0.    , 0.    , -0.009 , 0.009 ],
    [ 0.    ,  0.    , 0.    , 0.    , 0.    ]
])

Q_untrt = np.array([
    [-0.0085,  0.005,   0.0025, 0,       0.001],
    [0,       -0.014,   0.005,  0.004,   0.005],
    [0,        0,      -0.008,  0.003,   0.005],
    [0,        0,       0,     -0.009,   0.009],
    [0,        0,       0,      0,       0    ]
])

rng = np.random.default_rng(seed=2025)

def simulate_lifetime(Q, rng):
    """simulate one woman's lifetime (months) with treatment matrix Q"""

    #initial conditions
    state, t = 0, 0.0 # initial state is 1

    while state != 4:
        rate = -Q[state, state]
        t += rng.exponential(1 / rate) # exponential waiting time
        probs = Q[state].copy();  probs[state] = 0 # transition probabilitie, has to enter new state
        probs = - (probs / Q[state, state])
        state = rng.choice(5, p=probs)
    return t

N = 1_000
lifetimes_trt  = np.array([simulate_lifetime(Q_trt , rng) for _ in range(N)])
lifetimes_untr = np.array([simulate_lifetime(Q_untrt, rng) for _ in range(N)])

def km_step(lifetimes):
    """plotting KM curve"""
    times = np.sort(lifetimes)
    n = len(times)
    alive = n - np.arange(1, n+1) # survivors just after each death
    S_hat = alive / n
    # prepend t=0, S=1 to start the step plot
    x = np.concatenate(([0],  times))
    y = np.concatenate(([1.0], S_hat))
    return x, y

x_trt , y_trt  = km_step(lifetimes_trt)
x_untr, y_untr = km_step(lifetimes_untr)

#plot
plt.step(x_trt , y_trt , where='post', label='Preventive treatment')
plt.step(x_untr, y_untr, where='post', label='No treatment')
plt.xlabel('Months after surgery');  plt.ylabel('Kaplan-Meier Ŝ(t)')
plt.title('Survival curves: treated vs untreated (n = 1000 each)')
plt.legend();  plt.ylim(0, 1.01);  plt.grid(alpha=.3)
plt.show()