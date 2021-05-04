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

# Storing angular velocity trajectories and quaternion trajectories
w = []
q = []
odeq = []
odew = []


# Class for holding quaternion information
class Q:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __str__(self):
        return 'q1 = ' + str(self.x) + '\n' + 'q2 = ' + str(self.y) + '\n' + 'q3 = ' + str(
            self.z) + '\n' + 'q4 = ' + str(self.w) + '\n'


# Euler rate to angular vel. transformation matrix
def eulerToAngMat(phi, thet, psi):
    return np.matrix([[math.sin(thet) * math.sin(psi), math.cos(psi), 0],
                      [math.sin(thet) * math.cos(psi), -math.sin(psi), 0],
                      [math.cos(thet), 0, 1]])


# Attitude matrix (3x3) from euler angles
def eulerAttMat(phi, thet, psi):
    return np.matrix([[math.cos(psi) * math.cos(phi) - math.sin(psi) * math.cos(thet) * math.sin(phi),
                       math.cos(psi) * math.sin(phi) + math.sin(psi) * math.cos(thet) * math.cos(phi),
                       math.sin(phi) * math.sin(thet)],
                      [-math.cos(psi) * math.cos(thet) * math.sin(phi) - math.sin(psi) * math.cos(phi),
                       math.cos(psi) * math.cos(thet) * math.cos(phi) - math.sin(psi) * math.sin(phi),
                       math.cos(psi) * math.sin(thet)],
                      [math.sin(thet) * math.sin(phi), -math.sin(thet) * math.cos(phi), math.cos(thet)]])


# Create quaternions from euler rates
def eulerToQuat(phi, thet, psi):
    eulerMat = eulerAttMat(phi, thet, psi)
    q4 = 0.5 * math.sqrt(1 + eulerMat.item(0, 0) + eulerMat.item(1, 1) + eulerMat.item(2, 2))
    q1 = 1 / (4 * q4) * (eulerMat.item(1, 2) - eulerMat.item(2, 1))
    q2 = 1 / (4 * q4) * (eulerMat.item(2, 0) - eulerMat.item(0, 2))
    q3 = 1 / (4 * q4) * (eulerMat.item(0, 1) - eulerMat.item(1, 0))

    return Q(q1, q2, q3, q4)


# Returns the 'cross' (3x3) matrix of a (3x1) vector
def crossMat(alpha):
    return np.matrix([[0, -alpha.item(2), alpha.item(1)],
                      [alpha.item(2), 0, -alpha.item(0)],
                      [-alpha.item(1), alpha.item(0), 0]])


def quatProp(q, deltT, w):
    xi = 1 / (np.linalg.norm(w)) * (math.sin(0.5 * np.linalg.norm(w) * deltT) * w)
    cosVal = math.cos(0.5 * np.linalg.norm(w) * deltT)
    omega1 = np.subtract(cosVal * np.identity(3), crossMat(xi))
    omega2 = xi
    omega3 = -xi.T
    omega4 = np.matrix(cosVal)

    omega12 = np.concatenate((omega1, omega2), axis=1)
    omega34 = np.concatenate((omega3, omega4), axis=1)
    omega = np.concatenate((omega12, omega34), axis=0)

    return np.matmul(omega, q)


def rk4(q1, w1, h):
    L = np.zeros((3, 1))

    k1 = h * getF(q1, w1)
    l1 = h * getG(q1, w1, L)

    k2 = h * getF(q1 + 0.5 * k1, w1 + 0.5 * l1)
    l2 = h * getG(q1 + 0.5 * k1, w1 + 0.5 * l1, L)

    k3 = h * getF(q1 + 0.5 * k2, w1 + 0.5 * l2)
    l3 = h * getG(q1 + 0.5 * k2, w1 + 0.5 * l2, L)

    k4 = h * getF(q1 + k3, w1 + l3)
    l4 = h * getG(q1 + k3, w1 + l3, L)

    q2 = q1 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    w2 = w1 + (l1 + 2 * l2 + 2 * l3 + l4) / 6

    return [q2, w2]


def getF(q1, w1):
    rho = np.matrix([[q1.item(0)],
                     [q1.item(1)],
                     [q1.item(2)]])
    top = q1.item(3) * np.identity(3) + crossMat(rho)
    return 0.5 * np.matmul(np.concatenate((top, -rho.T), axis=0), w1)


def getG(q1, w1, L):
    part1 = np.matmul(np.matmul(np.matmul(-np.linalg.inv(J), crossMat(w1)), J), w1)
    part2 = np.matmul(np.linalg.inv(J), L)

    return part1 + part2


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

        xlist.append(e1 / (1 + e3))
        ylist.append(e2 / (1 + e3))

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

    w1 = []
    w2 = []
    w3 = []

    for t in range(1, 6000):
        phi += phidot * 0.1
        thet += thetdot * 0.1
        psi += psidot * 0.1

        omegas = np.matmul(eulerToAngMat(phi, thet, psi),
                           np.vstack(np.array([phidot * 0.1, thetdot * 0.1, psidot * 0.1])))

        # print(omegas)

        xlist.append(t * 0.1)

        w.append(omegas)

        w1.append(omegas.item(0))
        w2.append(omegas.item(1))
        w3.append(omegas.item(2))

    odew.append(w[0])
    print(w[0])

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
    # print(eulerMat0)
    q4 = 0.5 * math.sqrt(1 + eulerMat0.item(0, 0) + eulerMat0.item(1, 1) + eulerMat0.item(2, 2))
    q1 = 1 / (4 * q4) * (eulerMat0.item(1, 2) - eulerMat0.item(2, 1))
    q2 = 1 / (4 * q4) * (eulerMat0.item(2, 0) - eulerMat0.item(0, 2))
    q3 = 1 / (4 * q4) * (eulerMat0.item(0, 1) - eulerMat0.item(1, 0))

    xlist = []
    q1Lst = []
    q2Lst = []
    q3Lst = []
    q4Lst = []

    # print(Q(q1, q2, q3, q4))
    print('q1 = ' + str(q1) + '\n' + 'q2 = ' + str(q2) + '\n' + 'q3 = ' + str(q3) + '\n' + 'q4 = ' + str(q4) + '\n')

    oldq = np.matrix([[q1],
                      [q2],
                      [q3],
                      [q4]])

    odeq.append(oldq)

    for i in range(0, len(w)):
        newQ = quatProp(oldq, 0.1, w[i])
        q.append(newQ)
        q1Lst.append(newQ.item(0))
        q2Lst.append(newQ.item(1))
        q3Lst.append(newQ.item(2))
        q4Lst.append(newQ.item(3))
        xlist.append(i / 10)
        oldq = newQ

    # print(q1Lst)

    plt.plot(xlist, q1Lst, label='$q_1$')
    plt.plot(xlist, q2Lst, label='$q_2$')
    plt.plot(xlist, q3Lst, label='$q_3$')
    plt.plot(xlist, q4Lst, label='$q_4$')
    plt.legend()
    plt.xlabel('Time t [s]')
    plt.ylabel('Quaternion')
    plt.title('Quaternion Trajectory vs. Time')
    plt.show()


def q4():
    q1Lst, q2Lst, q3Lst, q4Lst = [], [], [], []

    # q1Lst = []
    # q2Lst = []
    # q3Lst = []
    # q4Lst = []
    w1Lst = []
    w2Lst = []
    w3Lst = []
    xlist = []

    for i in range(0, 6001):
        res = rk4(odeq[i], odew[i], 0.1)
        odeq.append(res[0])
        odew.append(res[1])

        q1Lst.append(res[0].item(0))
        q2Lst.append(res[0].item(1))
        q3Lst.append(res[0].item(2))
        q4Lst.append(res[0].item(3))

        w1Lst.append(res[1].item(0))
        w2Lst.append(res[1].item(1))
        w3Lst.append(res[1].item(2))

        xlist.append(i / 10)

    plt.plot(xlist, q1Lst, label='$q_1$')
    plt.plot(xlist, q2Lst, label='$q_2$')
    plt.plot(xlist, q3Lst, label='$q_3$')
    plt.plot(xlist, q4Lst, label='$q_4$')
    plt.legend()
    plt.xlabel('Time t [s]')
    plt.ylabel('Quaternion')
    plt.title('Runge Kutta Quaternion Trajectory vs. Time')
    plt.show()


q1()
q2()
q3()
q4()
