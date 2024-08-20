from . import *
import itertools

gate_map = {
    'X/2': 'halfX',
    'Y/2': 'halfY',
    'Z/2': 'halfZ',
    '-X/2': 'mhalfX',
    '-Y/2': 'mhalfY',
    '-Z/2': 'mhalfZ'
}

I2 = np.diag([1, 1]).astype(DTYPE)
sigmax = np.array([[0, 1], [1, 0]], dtype=DTYPE)
sigmay = np.array([[0, -1j], [1j, 0]], dtype=DTYPE)
sigmaz = np.array([[1, 0], [0, -1]], dtype=DTYPE)
H = np.array([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
X = -1j * sigmax
Y = -1j * sigmay
Z = -1j * sigmaz
mX = 1j * sigmax
mY = 1j * sigmay
mZ = 1j * sigmaz
halfX = (I2 + X) / 2**0.5
halfY = (I2 + Y) / 2**0.5
halfZ = (I2 + Z) / 2**0.5
mhalfX = (I2 - X) / 2**0.5
mhalfY = (I2 - Y) / 2**0.5
mhalfZ = (I2 - Z) / 2**0.5
halfXY = np.array(
    [[np.sqrt(2) / 2, -0.5 - 0.5j], [0.5 - 0.5j, np.sqrt(2) / 2]], dtype=DTYPE)
halfmXY = np.array(
    [[np.sqrt(2) / 2, -0.5 + 0.5j], [0.5 + 0.5j, np.sqrt(2) / 2]], dtype=DTYPE)
halfmXmY = np.array(
    [[np.sqrt(2) / 2, 0.5 + 0.5j], [-0.5 + 0.5j, np.sqrt(2) / 2]], dtype=DTYPE)
halfXmY = np.array(
    [[np.sqrt(2) / 2, 0.5 - 0.5j], [-0.5 - 0.5j, np.sqrt(2) / 2]], dtype=DTYPE)
CZ = np.diag([1, 1, 1, -1]).astype(DTYPE)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                dtype=DTYPE)

M00 = np.array([[1, 0], [0, 0]], dtype=DTYPE)
M01 = np.array([[0, 1], [0, 0]], dtype=DTYPE)


def rotation(alpha=0, theta=0, phi=0):
    '''
    auther: qguo
    update: 20170107

    Return the single qubit rotation operator matrix. Rotate the qubit 
    around (1, theta, phi) with angle alpha
    
    Parameters
    ----------
    alpha : rotation angle.
    theta : polar angle.
    phi : equatorial angle.
    '''

    nx = math.sin(theta) * math.cos(phi)
    ny = math.sin(theta) * math.sin(phi)
    nz = math.cos(theta)
    return rotation_angle_xyz(alpha, nx, ny, nz)


def rotation_angle_xyz(angle, nx, ny, nz):
    op = I2 * math.cos(angle / 2.0) - 1j * math.sin(
        angle / 2.0) * (nx * sigmax + ny * sigmay + nz * sigmaz)
    # op = scipy.linalg.expm(-1j * angle / 2 *
    #                        (nx * sigmax + ny * sigmay + nz * sigmaz))
    return op


_XEBops = {
    0: halfX,
    1: halfY,
    2: mhalfX,
    3: mhalfY,
    4: halfXY,
    5: halfmXY,
    6: halfmXmY,
    7: halfXmY
}

XEBops = {
    8 * i + j: rotation(i * np.pi / 4) @ _XEBops[j]
    for i, j in itertools.product(range(8), range(8))
}

# Clifford group for one qubit.
Clifford1 = {
    0: I2,
    1: X,
    2: Y,
    3: X @ Y,
    4: halfY @ halfX,
    5: mhalfY @ halfX,
    6: halfY @ mhalfX,
    7: mhalfY @ mhalfX,
    8: halfX @ halfY,
    9: mhalfX @ halfY,
    10: halfX @ mhalfY,
    11: mhalfX @ mhalfY,
    12: halfX,
    13: mhalfX,
    14: halfY,
    15: mhalfY,
    16: halfX @ halfY @ mhalfX,
    17: halfX @ mhalfY @ mhalfX,
    18: halfY @ X,
    19: mhalfY @ X,
    20: halfX @ Y,
    21: mhalfX @ Y,
    22: halfX @ halfY @ halfX,
    23: mhalfX @ halfY @ mhalfX
}


def qiskit_u(theta, phi, angle_lambda):
    '''
    Generic single-qubit rotation gate with 3 Euler angles.
    Rz(phi) * Ry(theta) * Rz(angle_lambda)
    '''
    return np.array([[
        math.cos(theta / 2), -np.exp(1j * angle_lambda) * math.sin(theta / 2)
    ],
                     [
                         np.exp(1j * phi) * math.sin(theta / 2),
                         np.exp(1j *
                                (phi + angle_lambda)) * math.cos(theta / 2)
                     ]],
                    dtype=DTYPE)
