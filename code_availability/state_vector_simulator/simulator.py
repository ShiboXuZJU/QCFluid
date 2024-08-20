from . import *
from . import Gate
from scipy.stats import unitary_group
from . import state_vector
from typing import Optional


def ptrace(probs, remain_q_idxs):
    '''
    ptrace prob with ptrace index
    probs.shape: (..., D)
    '''
    probs = np.asarray(probs)
    shape = list(probs.shape)
    D = shape[-1]
    _shape = shape[:-1]
    q_num = int(math.log2(D))
    return np.sum(
        np.moveaxis(probs.reshape(_shape + [2] * q_num),
                    np.asarray(remain_q_idxs) + len(_shape),
                    np.arange(len(remain_q_idxs)) +
                    len(_shape)).reshape(_shape + [2**len(remain_q_idxs), -1]),
        axis=-1)


def prob_to_expect(probs: np.ndarray) -> float:
    qnum = int(math.log2(len(probs)))
    factors = reduce(np.outer, [np.array([1.0, -1.0])] * qnum).reshape(-1)
    result = np.sum(factors * probs)
    return result


class Circuit(state_vector.ProductStateVector):
    def Pauli_expectation(self,
                          Pauli_str: str,
                          qubits: Optional[List[int]] = None) -> float:
        all_qubits = sorted(reduce(lambda x, y: x + y, self.qubits))
        if qubits is None:
            assert len(Pauli_str) == self.qnum
            qubits = all_qubits
        else:
            assert len(Pauli_str) == len(qubits)
        _copy = self.copy()
        remain_qubits = []
        for _qubit, _Pauli_str in zip(qubits, Pauli_str):
            if _Pauli_str == 'I':
                continue
            elif _Pauli_str == 'X':
                _copy.Ry(_qubit, -np.pi / 2)
                remain_qubits.append(_qubit)
            elif _Pauli_str == 'Y':
                _copy.Rx(_qubit, np.pi / 2)
                remain_qubits.append(_qubit)
            elif _Pauli_str == 'Z':
                remain_qubits.append(_qubit)
            else:
                raise ValueError(_Pauli_str)
        if len(remain_qubits) == 0:
            return 1.0
        probs = _copy.state_vector().probs(all_qubits)
        return prob_to_expect(
            ptrace(probs, [
                _idx for _idx, _qubit in enumerate(all_qubits)
                if _qubit in remain_qubits
            ]))

    def qiskit_u(self, qubit: int, theta, phi, angle_lambda):
        assert isinstance(qubit, int)
        self.apply_matrix([qubit], Gate.qiskit_u(theta, phi, angle_lambda))

    def rotation(self, qubit: int, alpha, theta, phi):
        assert isinstance(qubit, int)
        self.apply_matrix([qubit], Gate.rotation(alpha, theta, phi))

    def Rx(self, qubit: int, alpha):
        self.rotation(qubit, alpha, np.pi / 2, 0)

    def Ry(self, qubit: int, alpha):
        self.rotation(qubit, alpha, np.pi / 2, np.pi / 2)

    def Rz(self, qubit: int, alpha):
        self.rotation(qubit, alpha, 0, 0)

    def plus_gate(self, qubit: int, gate):
        assert isinstance(qubit, int)
        self.apply_matrix([qubit], getattr(Gate, Gate.gate_map.get(gate,
                                                                   gate)))

    def random_SU2(self, qubit: int):
        assert isinstance(qubit, int)
        self.apply_matrix([qubit], unitary_group.rvs(2))

    def Clifford1(self, qubit: int, Clifford_idx: int):
        assert isinstance(qubit, int)
        self.apply_matrix(qubit, Gate.Clifford1[Clifford_idx])

    def XEB_op(self, qubit: int, XEB_op_idx: int):
        assert isinstance(qubit, int)
        self.apply_matrix([qubit], Gate.XEBops[XEB_op_idx])

    def CZ(self, qubits: List[int]):
        assert isinstance(qubits, list)
        self.apply_matrix(qubits, Gate.CZ)

    def CNOT(self, qubits: List[int]):
        assert isinstance(qubits, list)
        self.apply_matrix(qubits, Gate.CNOT)

    def asymmetrical_depolarization_1q(self, qubit: int, p_X: float,
                                       p_Y: float, p_Z: float):
        assert isinstance(qubit, int)
        _r = np.random.rand()
        if _r < (1 - p_X - p_Y - p_Z):  # apply I
            pass
        elif _r < 1 - p_X - p_Y:  # apply Z
            self.apply_matrix([qubit], Gate.Z)
        elif _r < 1 - p_Y:  # apply X
            self.apply_matrix([qubit], Gate.X)
        else:  #apply Y
            self.apply_matrix([qubit], Gate.Y)

    def depolarization_1q(self, qubit: int, p: float):
        self.asymmetrical_depolarization_1q(qubit, p / 3, p / 3, p / 3)

    def phase_damping(self, qubit: int, gamma: float):
        '''
        Exponential-decay dephasing(T2), gamma=2*t_gate/T2
        Phase damping has exactly the same effect with phase flip.
        '''
        self.asymmetrical_depolarization_1q(qubit,
                                            p_X=0,
                                            p_Y=0,
                                            p_Z=(1 - (1 - gamma)**0.5) / 2)

    def amplitude_damping(self, qubit: int, gamma: float):
        '''
        Energy relaxation(T1), gamma=t_gate/T1
        '''
        _r = np.random.rand()
        _state_vector: state_vector.StateVector = self.state_vector([qubit])
        _state_vector.set_qubit_order([qubit])
        _state_tensor = _state_vector._state_tensor.reshape([2, -1])
        _P1 = np.sum(np.abs(_state_tensor[1])**2)
        if _r < gamma * _P1:
            _state_tensor[0] = _state_tensor[1] / _P1**0.5
            _state_tensor[1] = 0
        else:
            _state_tensor[1] *= (1 - gamma)**0.5
            _state_tensor /= (1 - _P1 * gamma)**0.5
        _state_vector._state_tensor = _state_tensor.reshape(
            _state_vector._shape)

    def asymmetrical_depolarization_2q(self, qubits: List[int], p_IX: float,
                                       p_IY: float, p_IZ: float, p_XI: float,
                                       p_XX: float, p_XY: float, p_XZ: float,
                                       p_YI: float, p_YX: float, p_YY: float,
                                       p_YZ: float, p_ZI: float, p_ZX: float,
                                       p_ZY: float, p_ZZ: float):
        assert isinstance(qubits, list)
        _r = np.random.rand()
        if _r < (1 -
                 (p_IX + p_IY + p_IZ + p_XI + p_XX + p_XY + p_XZ + p_YI +
                  p_YX + p_YY + p_YZ + p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply II
            pass
        elif _r < (1 - (p_IY + p_IZ + p_XI + p_XX + p_XY + p_XZ + p_YI + p_YX +
                        p_YY + p_YZ + p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply IX
            self.apply_matrix([qubits[1]], Gate.X)
        elif _r < (1 - (p_IZ + p_XI + p_XX + p_XY + p_XZ + p_YI + p_YX + p_YY +
                        p_YZ + p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply IY
            self.apply_matrix([qubits[1]], Gate.Y)
        elif _r < (1 - (p_XI + p_XX + p_XY + p_XZ + p_YI + p_YX + p_YY + p_YZ +
                        p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply IZ
            self.apply_matrix([qubits[1]], Gate.Z)
        elif _r < (1 - (p_XX + p_XY + p_XZ + p_YI + p_YX + p_YY + p_YZ + p_ZI +
                        p_ZX + p_ZY + p_ZZ)):  # apply XI
            self.apply_matrix([qubits[0]], Gate.X)
        elif _r < (1 - (p_XY + p_XZ + p_YI + p_YX + p_YY + p_YZ + p_ZI + p_ZX +
                        p_ZY + p_ZZ)):  # apply XX
            self.apply_matrix([qubits[0]], Gate.X)
            self.apply_matrix([qubits[1]], Gate.X)
        elif _r < (1 - (p_XZ + p_YI + p_YX + p_YY + p_YZ + p_ZI + p_ZX + p_ZY +
                        p_ZZ)):  # apply XY
            self.apply_matrix([qubits[0]], Gate.X)
            self.apply_matrix([qubits[1]], Gate.Y)
        elif _r < (1 - (p_YI + p_YX + p_YY + p_YZ + p_ZI + p_ZX + p_ZY +
                        p_ZZ)):  # apply XZ
            self.apply_matrix([qubits[0]], Gate.X)
            self.apply_matrix([qubits[1]], Gate.Z)
        elif _r < (
                1 -
            (p_YX + p_YY + p_YZ + p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply YI
            self.apply_matrix([qubits[0]], Gate.Y)
        elif _r < (1 - (p_YY + p_YZ + p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply YX
            self.apply_matrix([qubits[0]], Gate.Y)
            self.apply_matrix([qubits[1]], Gate.X)
        elif _r < (1 - (p_YZ + p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply YY
            self.apply_matrix([qubits[0]], Gate.Y)
            self.apply_matrix([qubits[1]], Gate.Y)
        elif _r < (1 - (p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply YZ
            self.apply_matrix([qubits[0]], Gate.Y)
            self.apply_matrix([qubits[1]], Gate.Z)
        elif _r < (1 - (p_ZX + p_ZY + p_ZZ)):  # apply ZI
            self.apply_matrix([qubits[0]], Gate.Z)
        elif _r < (1 - (p_ZY + p_ZZ)):  # apply ZX
            self.apply_matrix([qubits[0]], Gate.Z)
            self.apply_matrix([qubits[1]], Gate.X)
        elif _r < (1 - (p_ZZ)):  # apply ZY
            self.apply_matrix([qubits[0]], Gate.Z)
            self.apply_matrix([qubits[1]], Gate.Y)
        else:  # apply ZZ
            self.apply_matrix([qubits[0]], Gate.Z)
            self.apply_matrix([qubits[1]], Gate.Z)

    def depolarization_2q(self, qubits: List[int], p: float):
        self.asymmetrical_depolarization_2q(qubits, *([p / 15] * 15))
