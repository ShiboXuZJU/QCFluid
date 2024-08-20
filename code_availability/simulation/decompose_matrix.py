import numpy as np
import math
import itertools


NP_DTYPE = np.complex64

sigmaI = np.eye(2, dtype=NP_DTYPE)
sigmaX = np.array([[0, 1], [1, 0]], dtype=NP_DTYPE)
sigmaY = np.array([[0, -1j], [1j, 0]], dtype=NP_DTYPE)
sigmaZ = np.array([[1, 0], [0, -1]], dtype=NP_DTYPE)


def decompose(op: np.ndarray) -> dict:
    '''
    matrix dimension should not greater than 2^14
    ------
    return
    coe_matrix: [4, 4, ...]
        -value: tr(PauliString @ op)
    order: IXYZ
    '''
    q_num = int(math.log2(op.shape[0]))
    if q_num >= 14:
        raise Exception('qubit number too large, memory consumption too large')
    Pauli_basis = np.asarray([sigmaI, sigmaX, sigmaY, sigmaZ])
    op = op.reshape([2] * q_num * 2)
    op = np.transpose(op, np.hstack([[i, i + q_num] for i in range(q_num)]))
    for _ in range(q_num):
        op = np.einsum('abc,dba->cd', op.reshape([2, 2, -1]), Pauli_basis)
    return {
        ''.join(k): v
        for k, v in zip(list(itertools.product('IXYZ', repeat=q_num)),
                        op.reshape(-1))
    }