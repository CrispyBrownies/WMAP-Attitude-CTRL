# Stephen Chen
# MAE425: Space Dynamics and Control
# Created: 4/23/2021

import math
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import rc
#
# rc('text', usetex = True)


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

# Storing angular velocity trajectories for Q2
w1 = []
w2 = []
w3 = []


# Class for holding quaternion information
class Q:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __str__(self):
        return 'q1 = ' + str(self.x) + '\n' + 'q2 = ' + str(self.y) + '\n' + 'q3 = ' + str(self.z) + '\n' + 'q4 = ' + str(self.w) + '\n'


# Euler rate to angular vel. transformation matrix
def eulerToAngMat(phi, thet, psi):
    return np.matrix([[math.sin(thet)*math.sin(psi), math.cos(psi), 0],
                      [math.sin(thet)*math.cos(psi), -math.sin(psi), 0],
                      [math.cos(thet), 0, 1]])


# Attitude matrix (3x3) from euler angles
def eulerAttMat(phi, thet, psi):
    return np.matrix([[math.cos(psi)*math.cos(phi)-math.sin(psi)*math.cos(thet)*math.sin(phi), math.cos(psi)*math.sin(phi)+math.sin(psi)*math.cos(thet)*math.cos(phi), math.sin(phi)*math.sin(thet)],
                     [-math.cos(psi)*math.cos(thet)*math.sin(phi)-math.sin(psi)*math.cos(phi), math.cos(psi)*math.cos(thet)*math.cos(phi)-math.sin(psi)*math.sin(phi), math.cos(psi)*math.sin(thet)],
                     [math.sin(thet)*math.sin(phi), -math.sin(thet)*math.cos(phi), math.cos(thet)]])


# Create quaternions from euler rates
def eulerToQuat(phi, thet, psi):
    eulerMat = eulerAttMat(phi, thet, psi)
    q4 = 0.5*math.sqrt(1+eulerMat.item(0, 0)+eulerMat.item(1, 1)+eulerMat.item(2, 2))
    q1 = 1/(4*q4)*(eulerMat.item(1, 2)-eulerMat.item(2, 1))
    q2 = 1/(4*q4)*(eulerMat.item(2, 0)-eulerMat.item(0, 2))
    q3 = 1/(4*q4)*(eulerMat.item(0, 1)-eulerMat.item(1, 0))

    return Q(q1, q2, q3, q4)


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
    plt.title('Desired Scan Pattern X/Y')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def q2():
    phi = phi0
    thet = thet0
    psi = psi0

    xlist = []

    for t in range(1, 6000):
        phi += phidot*0.1
        thet += thetdot*0.1
        psi += psidot*0.1

        omegas = np.matmul(eulerToAngMat(phi, thet, psi), np.vstack(np.array([phidot*0.1, thetdot*0.1, psidot*0.1])))

        xlist.append(t*0.1)

        w1.append(omegas.item(0))
        w2.append(omegas.item(1))
        w3.append(omegas.item(2))

    plt.plot(xlist, w1, label='$\omega_1$(t)')
    plt.plot(xlist, w2, label='$\omega_2$(t)')
    plt.plot(xlist, w3, label='$\omega_3$(t)')
    plt.title('WMAP Angular Velocities vs. Time')
    plt.xlabel('Time t [s]')
    plt.ylabel('Angular Velocity $\omega$(t) [rad/s]')
    plt.legend()
    plt.show()


def q3():
    eulerMat0 = eulerAttMat(phi0, thet0, psi0)
    print(eulerMat0)
    q4 = 0.5 * math.sqrt(1 + eulerMat0.item(0, 0) + eulerMat0.item(1, 1) + eulerMat0.item(2, 2))
    q1 = 1 / (4 * q4) * (eulerMat0.item(1, 2) - eulerMat0.item(2, 1))
    q2 = 1 / (4 * q4) * (eulerMat0.item(2, 0) - eulerMat0.item(0, 2))
    q3 = 1 / (4 * q4) * (eulerMat0.item(0, 1) - eulerMat0.item(1, 0))

    print(Q(q1, q2, q3, q4))



q3()



