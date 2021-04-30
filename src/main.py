# Stephen Chen
# MAE425: Space Dynamics and Control
# Created: 4/23/2021

import math
import numpy as np
import matplotlib.pyplot as plt

# Euler Angles and Euler Rates (3-1-3)
phi0 = 0  # [rad]
thet0 = 0.3927  # [rad]
psi0 = 0  # [rad]

phidot = 0.001745  # [rad/s]
thetdot = 0  # [rad/s]
psidot = 0.04859  # [rad/s]

# Inertia Matrix
J = np.matrix([[399, -2.71, -1.21],  # [kg-m^2]
               [-2.71, 377, 2.14],
               [-1.21, 2.14, 377]])


# Calculating desired scan pattern from t=1s to t=7200s
def q1():
    phi = phi0
    thet = thet0
    psi = psi0

    xlist = []
    ylist = []

    for t in range(1, 7200):
        phi += phidot
        thet += thetdot
        psi += psidot

        e1 = math.cos(thet) * math.cos(psi) * math.cos(phi) - math.pow(math.cos(thet), 2) * math.sin(psi) * math.sin(
            phi) + math.pow(math.sin(thet), 2) * math.sin(phi)
        e2 = math.cos(thet) * math.cos(psi) * math.sin(phi) + math.pow(math.cos(thet), 2) * math.sin(psi) * math.cos(
            phi) - math.pow(math.sin(thet), 2) * math.cos(phi)
        e3 = math.cos(thet) * math.sin(thet) * (math.sin(psi) + 1)

        xlist.append(e1/(1+e3))
        ylist.append(e2/(1+e3))

    plt.plot(xlist, ylist, linewidth=1.0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def q2():
    phi = phi0
    thet = thet0
    psi = psi0

    for t in range(1,6000):
        phi += phidot
        thet += thetdot
        psi += psidot

q1()



