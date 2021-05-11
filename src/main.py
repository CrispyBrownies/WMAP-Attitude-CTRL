# Stephen Chen
# MAE425: Space Dynamics and Control
# Created: 4/23/2021

import math
import numpy as np
import matplotlib.pyplot as plt

# Change path to save image elsewhere
path = "../Figures/"

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
# odeq = []
# odew = []
q0 = np.matrix([[0], [0], [0], [0]])
w0 = np.matrix([[0], [0], [0]])
l1Lst, l2Lst, l3Lst = [], [], []

compFig = 5
l1Mat, l2Mat, l3Mat = {}, {}, {}
w1Mat, w2Mat, w3Mat = {}, {}, {}
q1Mat, q2Mat, q3Mat = {}, {}, {}

# Set showControl to False to see quaternion trajectory without control law L(t)
showControl = False


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


# Propagates the quaternion equations forward in time using q(t+deltT) = omega[w(t)]*q(t)
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


# Numerically integrate next q and w value using 4th order Runge Kutta Routine
def rk4(q1, w1, qt, wt, k, h):
    rho = np.matrix([[qt.item(0)],
                     [qt.item(1)],
                     [qt.item(2)]])
    top = qt.item(3) * np.identity(3) + crossMat(rho)
    xi = np.concatenate((top, -rho.T), axis=0)

    Lmat = np.matmul(-k * xi.T, q1) - k * (w1 - wt)

    l1Lst.append(Lmat.item(0))
    l2Lst.append(Lmat.item(1))
    l3Lst.append(Lmat.item(2))

    k1 = h * getF(q1, w1)
    l1 = h * getG(q1, w1, Lmat)

    k2 = h * getF(q1 + 0.5 * k1, w1 + 0.5 * l1)
    l2 = h * getG(q1 + 0.5 * k1, w1 + 0.5 * l1, Lmat)

    k3 = h * getF(q1 + 0.5 * k2, w1 + 0.5 * l2)
    l3 = h * getG(q1 + 0.5 * k2, w1 + 0.5 * l2, Lmat)

    k4 = h * getF(q1 + k3, w1 + l3)
    l4 = h * getG(q1 + k3, w1 + l3, Lmat)

    q2 = q1 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    w2 = w1 + (l1 + 2 * l2 + 2 * l3 + l4) / 6

    return [q2, w2]


# Calculates the "f" part of Runge Kutta
def getF(q1, w1):
    rho = np.matrix([[q1.item(0)],
                     [q1.item(1)],
                     [q1.item(2)]])
    top = q1.item(3) * np.identity(3) + crossMat(rho)
    return 0.5 * np.matmul(np.concatenate((top, -rho.T), axis=0), w1)


# Calculates the "g" part of Runge Kutta
def getG(q1, w1, L):
    part1 = np.matmul(np.matmul(np.matmul(-np.linalg.inv(J), crossMat(w1)), J), w1)
    part2 = np.matmul(np.linalg.inv(J), L)
    # print(part1)
    # print(part2)

    return part1 + part2


# Function to multiple two quaternions
def multQuat(q1, q2):
    neww = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    newx = q1.x * q2.w + q1.w * q2.x - q1.z * q2.y + q1.y * q2.z
    newy = q1.y * q2.w + q1.z * q2.x + q1.w * q2.y - q1.x * q2.z
    newz = q1.z * q2.w - q1.y * q2.x + q1.x * q2.y + q1.w * q2.z

    return Q(newx, newy, newz, neww)


# Function to calculate the error quaternion between two input quaternions
def getQuatDiff(q1, q2):
    q2conj = Q(-q2.x, -q2.y, -q2.z, q2.w)
    return multQuat(q1, q2conj)


# Turns a numpy array into a Quaternion object
def toQuat(array):
    return Q(array.item(0), array.item(1), array.item(2), array.item(3))


# Calculating desired scan pattern from t=0s to t=7200s
def q1():

    # Setting initial conditions
    phi = phi0
    thet = thet0
    psi = psi0

    xlist = []
    ylist = []

    # Calculating the scan pattern over the time-span by changing Euler angles by Euler rates
    for t in range(0, 7200):
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

    # Plotting the scan pattern
    plt.figure(1)
    plt.plot(xlist, ylist, linewidth=1.0)
    plt.title('Desired Scan Pattern X/Y')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig(path + "ScanPattern")


# Calculate body-fixed angular velocity trajectory from t=0s to t=600s
def q2():

    global w0

    # Setting IC
    phi = phi0
    thet = thet0
    psi = psi0

    xlist = []

    w1 = []
    w2 = []
    w3 = []

    # Calculate angular velocity of satellite through time-span
    for t in range(0, 6000):
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

    w0 = w[0]

    # Plotting angular velocity of satellite
    plt.figure(2)
    plt.plot(xlist, w1, label='$\omega_1$(t)')
    plt.plot(xlist, w2, label='$\omega_2$(t)')
    plt.plot(xlist, w3, label='$\omega_3$(t)')
    plt.title('WMAP Angular Velocities vs. Time')
    plt.xlabel('Time t [s]')
    plt.ylabel('Angular Velocity $\omega$(t) [rad/s]')
    plt.legend()
    # plt.savefig(path + "WMAPAngVel")


# Calculate the quaternion trajectory using attitude matrix (see Appendix for derivation)
def q3():

    global q0

    # Generate attitude matrix and get initial quaternion
    eulerMat0 = eulerAttMat(phi0, thet0, psi0)
    q4 = 0.5 * math.sqrt(1 + eulerMat0.item(0, 0) + eulerMat0.item(1, 1) + eulerMat0.item(2, 2))
    q1 = 1 / (4 * q4) * (eulerMat0.item(1, 2) - eulerMat0.item(2, 1))
    q2 = 1 / (4 * q4) * (eulerMat0.item(2, 0) - eulerMat0.item(0, 2))
    q3 = 1 / (4 * q4) * (eulerMat0.item(0, 1) - eulerMat0.item(1, 0))

    print("Initial quaternion is: ")
    print(Q(q1, q2, q3, q4))

    oldq = np.matrix([[q1],
                      [q2],
                      [q3],
                      [q4]])

    q0 = oldq

    xlist = []
    q1Lst, q2Lst, q3Lst, q4Lst = [], [], [], []

    # Calculate quaternion trajectory using previous angular velocity trajectory
    for i in range(0, len(w)):
        newQ = quatProp(oldq, 1, w[i])
        q.append(newQ)
        q1Lst.append(newQ.item(0))
        q2Lst.append(newQ.item(1))
        q3Lst.append(newQ.item(2))
        q4Lst.append(newQ.item(3))
        xlist.append(i / 10)
        oldq = newQ

    # Plotting quaternion trajectory
    plt.figure(3)
    plt.plot(xlist, q1Lst, label='$q_1$')
    plt.plot(xlist, q2Lst, label='$q_2$')
    plt.plot(xlist, q3Lst, label='$q_3$')
    plt.plot(xlist, q4Lst, label='$q_4$')
    plt.legend()
    plt.xlabel('Time t [s]')
    plt.ylabel('Quaternion')
    plt.title('Quaternion Trajectory vs. Time')
    # plt.savefig(path + "QuatTrajectory")


# Using 4th Order Runge Kutta routine, numerically integrate quaternion trajectory,
def q4(k, ec):
    global showControl
    global compFig

    odew = []
    odeq = []

    odeq.append(q0)
    odew.append(w0)

    q1Lst, q2Lst, q3Lst, q4Lst = [], [], [], []
    w1Lst, w2Lst, w3Lst = [], [], []

    xlist = []
    q1diff, q2diff, q3diff = [], [], []
    w1diff, w2diff, w3diff = [], [], []

    if showControl:
        odeq[0] = np.matrix([[0], [0], [0], [1]])
        odew[0] = np.matrix([[0], [0], [0]])

    # Numerically integrate quat. trajectory
    for i in range(0, 6000):
        res = rk4(odeq[i], odew[i], q[i], w[i], k, 1)

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

    # Plotting the RK4 quaternion trajectory
    if not ec:
        plt.figure(4)
        plt.plot(xlist, q1Lst, label='$q_1$')
        plt.plot(xlist, q2Lst, label='$q_2$')
        plt.plot(xlist, q3Lst, label='$q_3$')
        plt.plot(xlist, q4Lst, label='$q_4$')
        plt.legend()
        plt.xlabel('Time t [s]')
        plt.ylabel('Quaternion')
        plt.title('Runge Kutta Quaternion k = '+str(k)+' Trajectory vs. Time')
        # plt.xlim([0, 60])
        # plt.savefig(path + "RK4QuatTrajectoryk="+str(k))

    if showControl:
        for i in range(0, 6000):
            q1 = odeq[i]
            q2 = q[i]
            w1 = odew[i]
            w2 = w[i]

            diffQ = getQuatDiff(toQuat(q1), toQuat(q2))
            q1diff.append(0.5 * diffQ.x * 180 / math.pi)
            q2diff.append(0.5 * diffQ.y * 180 / math.pi)
            q3diff.append(0.5 * diffQ.z * 180 / math.pi)

            w1diff.append(w2.item(0) - w1.item(0))
            w2diff.append(w2.item(1) - w1.item(1))
            w3diff.append(w2.item(2) - w1.item(2))

        q1Mat[k] = q1diff
        q2Mat[k] = q2diff
        q3Mat[k] = q3diff

        w1Mat[k] = w1diff
        w2Mat[k] = w2diff
        w3Mat[k] = w3diff

        l1Mat[k] = l1Lst
        l2Mat[k] = l2Lst
        l3Mat[k] = l3Lst


# Determining how control law and error changes with different control gain kq and kw values
def ec():

    global compFig

    krange = [0, 5, 10, 20]
    xlist = []
    figLst = []

    for x in range(0, 6000):
        xlist.append(x*0.1)

    figLst.append(plt.subplots(3, sharex=True))
    figLst.append(plt.subplots(3, sharex=True))
    figLst.append(plt.subplots(3, sharex=True))

    for k in krange:
        global l1Lst, l2Lst, l3Lst
        l1Lst, l2Lst, l3Lst = [], [], []
        q4(k, True)

        # Plotting comparison graphs for different k values
        ((figLst[0])[1])[0].plot(xlist, q1Mat[k], '-', label='k = ' + str(k))
        ((figLst[0])[1])[1].plot(xlist, q2Mat[k], '-', label='k = ' + str(k))
        ((figLst[0])[1])[2].plot(xlist, q3Mat[k], '-', label='k = ' + str(k))
        (figLst[0])[0].suptitle('Quaternion Error vs. Time')
        ((figLst[0])[1])[0].legend(loc=1)
        (figLst[0])[0].text(0.5, 0.04, 'Time t [s]', ha='center')
        (figLst[0])[0].text(0.04, 0.5, 'Quaternion Error [deg]', va='center', rotation='vertical')
        # ((figLst[0])[1])[0].axis(xmin=0, xmax=60)
        # ((figLst[0])[1])[1].axis(xmin=0, xmax=60)
        # ((figLst[0])[1])[2].axis(xmin=0, xmax=60)

        ((figLst[1])[1])[0].plot(xlist, w1Mat[k], '-', label='k = ' + str(k))
        ((figLst[1])[1])[1].plot(xlist, w2Mat[k], '-', label='k = ' + str(k))
        ((figLst[1])[1])[2].plot(xlist, w3Mat[k], '-', label='k = ' + str(k))
        (figLst[1])[0].suptitle('Angular Velocity Error vs. Time')
        ((figLst[1])[1])[0].legend(loc=1)
        (figLst[1])[0].text(0.5, 0.04, 'Time t [s]', ha='center')
        (figLst[1])[0].text(0.04, 0.5, 'Angular Velocity Error [rad/s]', va='center', rotation='vertical')
        ((figLst[1])[1])[0].axis(xmin=0, xmax=60)
        ((figLst[1])[1])[1].axis(xmin=0, xmax=60)
        ((figLst[1])[1])[2].axis(xmin=0, xmax=60)

        ((figLst[2])[1])[0].plot(xlist, l1Mat[k], '-', label='k = ' + str(k))
        ((figLst[2])[1])[1].plot(xlist, l2Mat[k], '-', label='k = ' + str(k))
        ((figLst[2])[1])[2].plot(xlist, l3Mat[k], '-', label='k = ' + str(k))
        (figLst[2])[0].suptitle('Control Law vs. Time')
        ((figLst[2])[1])[0].legend(loc=1)
        (figLst[2])[0].text(0.5, 0.04, 'Time t [s]', ha='center')
        (figLst[2])[0].text(0.04, 0.5, 'Control Law [rad/s]', va='center', rotation='vertical')
        # ((figLst[2])[1])[0].axis(xmin=0, xmax=60)
        # ((figLst[2])[1])[1].axis(xmin=0, xmax=60)
        # ((figLst[2])[1])[2].axis(xmin=0, xmax=60)

    # (figLst[0])[0].savefig(path + "QuatErrorComp")
    # (figLst[1])[0].savefig(path + "AngVelErrorComp")
    # (figLst[2])[0].savefig(path + "ControlLawComp")


q1()
q2()
q3()
q4(0, False)
# ec()  # Enable to see comparison, will take longer and generate more plots
plt.show()