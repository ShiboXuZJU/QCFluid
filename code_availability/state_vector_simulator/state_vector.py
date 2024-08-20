from . import *


class StateVector:
    def __init__(self,
                 qubits: List[int],
                 initial_state: np.ndarray = None) -> None:
        assert len(np.unique(qubits)) == len(
            qubits), f"[confilct]qubits{qubits} have same label"
        self.qubits = qubits
        self.qnum = len(self.qubits)
        self._shape = [2] * self.qnum
        if initial_state is None:
            self._state_tensor = np.zeros(2**self.qnum, dtype=DTYPE)
            self._state_tensor[0] = 1
            self._state_tensor = np.reshape(self._state_tensor, self._shape)
        else:
            if np.any(np.array(np.shape(initial_state)) != 2):
                self._state_tensor = np.reshape(initial_state, self._shape)
            else:
                self._state_tensor = initial_state

    def __repr__(self) -> str:
        return '<' + self.__class__.__name__ + '> ' + f"qubits:{self.qubits}"

    def copy(self):
        return self.__class__(self.qubits.copy(), self._state_tensor.copy())

    def q_idxs(self, qubits: Sequence[int]) -> np.ndarray:
        return np.array([self.qubits.index(_qubit) for _qubit in qubits])

    def set_qubit_order(self, ordered_qubits: Sequence[int]) -> None:
        q_idxs = self.q_idxs(ordered_qubits)
        if np.any(np.arange(len(q_idxs)) != q_idxs):
            self._state_tensor = np.moveaxis(self._state_tensor, q_idxs,
                                             range(len(q_idxs)))
            self.qubits = ordered_qubits + [
                _qubit
                for _qubit in self.qubits if _qubit not in ordered_qubits
            ]

    def apply_matrix(self, qubits: List[int], matrix: np.ndarray) -> None:
        self.set_qubit_order(qubits)
        self._state_tensor = np.reshape(self._state_tensor,
                                        [2**len(qubits), -1])
        self._state_tensor = np.dot(matrix, self._state_tensor)
        self._state_tensor = np.reshape(self._state_tensor, self._shape)

    def density_matrix_of(self, qubits: List[int]) -> np.ndarray:
        self.set_qubit_order(qubits)
        state = np.reshape(self._state_tensor, [2**len(qubits), -1])
        return np.dot(state, state.conj().T)

    def state_vector(self, ordered_qubits: List[int] = None) -> np.ndarray:
        if ordered_qubits is not None:
            self.set_qubit_order(ordered_qubits)
        return np.reshape(self._state_tensor, [-1])

    def probs(self, ordered_qubits: List[int] = None) -> np.ndarray:
        return np.round(
            np.abs(self.state_vector(ordered_qubits))**2, PRECISION)

    def counts(self, ordered_qubits: List[int] = None) -> dict:
        probs = self.probs(ordered_qubits)
        mask = probs != 0
        counts = dict(zip(np.arange(len(probs))[mask], probs[mask]))
        return {
            np.binary_repr(key, self.qnum): value
            for key, value in counts.items()
        }


class ProductStateVector:
    def __init__(
            self,
            non_entangled_state_vectors: list[StateVector] = None) -> None:
        if non_entangled_state_vectors is None:
            self._state_vectors = []
        else:
            self._state_vectors = non_entangled_state_vectors

    @property
    def qubits(self) -> List[List[int]]:
        if len(self._state_vectors) == 0:
            return [[]]
        else:
            return [
                _state_vector.qubits for _state_vector in self._state_vectors
            ]

    @property
    def qnum(self) -> int:
        if len(self._state_vectors) == 0:
            return 0
        else:
            return np.sum(
                [_state_vector.qnum for _state_vector in self._state_vectors])

    def __repr__(self) -> str:
        return '<' + self.__class__.__name__ + '> ' + f"qubits:{self.qubits}"

    def copy(self):
        return self.__class__(
            [_state_vector.copy() for _state_vector in self._state_vectors])

    def _update_state_vector(self, qubits: Sequence[int]) -> None:
        _existed_qubits = self.qubits
        _existed_qubits_all = reduce(lambda x, y: x + y, _existed_qubits)
        for _qubit in qubits:
            if _qubit not in _existed_qubits_all:
                self._state_vectors.append(StateVector([_qubit]))

    def _merge_state_vector(self,
                            qubits: Sequence[int],
                            replace=True) -> StateVector:
        self._update_state_vector(qubits)
        _merge_state_vector_idx = set()
        for _qubit in qubits:
            for idx, _state_vector in enumerate(self._state_vectors):
                if _qubit in _state_vector.qubits:
                    _merge_state_vector_idx.add(idx)
        if len(_merge_state_vector_idx) > 1:
            _merged_qubits = []
            _merged_state_tensor = None
            for idx in _merge_state_vector_idx:
                _state_vector: StateVector = self._state_vectors[idx]
                _merged_qubits += _state_vector.qubits
                if _merged_state_tensor is None:
                    _merged_state_tensor = _state_vector.state_vector()
                else:
                    _merged_state_tensor = np.outer(
                        _merged_state_tensor, _state_vector.state_vector())
            _merged_state_vector = StateVector(_merged_qubits,
                                               _merged_state_tensor)
            if replace:
                self._state_vectors = [
                    _state_vector
                    for idx, _state_vector in enumerate(self._state_vectors)
                    if idx not in _merge_state_vector_idx
                ] + [_merged_state_vector]
            return _merged_state_vector
        else:
            return self._state_vectors[_merge_state_vector_idx.pop()]

    def apply_matrix(self, qubits: Sequence[int], matrix: np.ndarray):
        self._merge_state_vector(qubits,
                                 replace=True).apply_matrix(qubits, matrix)

    def state_vector(self, qubits: Sequence[int] = None) -> StateVector:
        qubits = reduce(lambda x, y: x + y,
                        self.qubits) if qubits is None else qubits
        return self._merge_state_vector(qubits, replace=False)

    def density_matrix_of(self, qubits: Sequence[int]) -> np.ndarray:
        return self.state_vector(qubits).density_matrix_of(qubits)
