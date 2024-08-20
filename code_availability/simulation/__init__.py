import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import pathlib
import sys
import math
import pickle
from functools import reduce
import tqdm

from state_vector_simulator import simulator
from . import decompose_matrix

root = pathlib.Path(__file__).parent
if not root.joinpath('temp').exists():
    root.joinpath('temp').mkdir()


def get_regex(op):
    pattern = []
    for i in op:
        if i in 'XY':
            pattern.append(i)
        elif i == 'Z':
            pattern.append('[IZ]')
        elif i == 'I':
            pattern.append('[IXYZ]')
        else:
            raise RuntimeError(op + ' exist unknown character ' + i)
    return re.compile(''.join(pattern))


def current_matrix(N, x, y, periodic=False):
    """
    x->column;y->row
    generate current_matrix in (x,y)

    N:qubit number in 1 dimention
    x:[0,2**N-1]
    y:[0,2**N-1]
    periodic: True if using periodic bond condition
    """

    jx = np.zeros((2**(2 * N), 2**(2 * N)), dtype=np.complex128)
    jy = np.zeros((2**(2 * N), 2**(2 * N)), dtype=np.complex128)
    delta = 2 * np.pi / 2**N
    element = 1 / (4j * delta)
    if periodic:
        jy[y * 2**N + x][((y + 1) * 2**N + x) % (2**(2 * N))] = element
        jy[y * 2**N + x][((y - 1) * 2**N + x) % (2**(2 * N))] = -element
        jy[((y - 1) * 2**N + x) % (2**(2 * N))][y * 2**N + x] = element
        jy[((y + 1) * 2**N + x) % (2**(2 * N))][y * 2**N + x] = -element

        jx[y * 2**N + x][(y * 2**N + x + 1) % (2**(2 * N))] = element
        jx[y * 2**N + x][(y * 2**N + x - 1) % (2**(2 * N))] = -element
        jx[(y * 2**N + x - 1) % (2**(2 * N))][y * 2**N + x] = element
        jx[(y * 2**N + x + 1) % (2**(2 * N))][y * 2**N + x] = -element
    else:
        if y == 2**N - 1:
            jy[y * 2**N + x][(y - 1) * 2**N + x] = -element * 2
            jy[(y - 1) * 2**N + x][y * 2**N + x] = element * 2
        elif y == 0:
            jy[y * 2**N + x][(y + 1) * 2**N + x] = element * 2
            jy[(y + 1) * 2**N + x][y * 2**N + x] = -element * 2
        else:
            jy[y * 2**N + x][(y + 1) * 2**N + x] = element
            jy[(y + 1) * 2**N + x][y * 2**N + x] = -element
            jy[y * 2**N + x][(y - 1) * 2**N + x] = -element
            jy[(y - 1) * 2**N + x][y * 2**N + x] = element

        if x == 2**N - 1:
            jx[y * 2**N + x][y * 2**N + x - 1] = -element * 2
            jx[y * 2**N + x - 1][y * 2**N + x] = element * 2
        elif x == 0:
            jx[y * 2**N + x][y * 2**N + x + 1] = element * 2
            jx[y * 2**N + x + 1][y * 2**N + x] = -element * 2
        else:
            jx[y * 2**N + x][y * 2**N + x + 1] = element
            jx[y * 2**N + x + 1][y * 2**N + x] = -element
            jx[y * 2**N + x][y * 2**N + x - 1] = -element
            jx[y * 2**N + x - 1][y * 2**N + x] = element

    return jx, jy


def dump_decomposed_current_matrix(threshold=1e-6):
    print(
        '[dump_decomposed_current_matrix]: Calculating... (This process may cost half an hour)'
    )
    N = 2**5
    jx_coeff = {}
    jy_coeff = {}
    for _x in tqdm.tqdm(range(N), desc='x'):
        for _y in tqdm.tqdm(range(N), desc='y'):
            jx, jy = current_matrix(5, _x, _y)
            _jx_coeff = decompose_matrix.decompose(
                jx)  # tr(j_x @ Pauli string)
            _jy_coeff = decompose_matrix.decompose(jy)
            jx_coeff[(_x, _y)] = {
                k: v / 2**10
                for k, v in _jx_coeff.items() if abs(v) > threshold
            }
            jy_coeff[(_x, _y)] = {
                k: v / 2**10
                for k, v in _jy_coeff.items() if abs(v) > threshold
            }
    with open(root.joinpath('temp/decomposed_current_matrix.pkl'), 'wb') as f:
        pickle.dump({'jx': jx_coeff, 'jy': jy_coeff}, f)


def load_decomposed_current_matrix():
    with open(root.joinpath('temp/decomposed_current_matrix.pkl'), 'rb') as f:
        result = pickle.load(f)
    return result


if not root.joinpath('temp/decomposed_current_matrix.pkl').exists():
    dump_decomposed_current_matrix()
DECOMPOSED_CURRENT_MATRIX = load_decomposed_current_matrix()


def dump_sampling_op_info():
    print('[dump_sampling_op_info]: Calculating...')
    result = DECOMPOSED_CURRENT_MATRIX
    jx = result['jx']
    jy = result['jy']
    Pauli_string = {}
    for v in jx.values():
        Pauli_string.update(v)
    for v in jy.values():
        Pauli_string.update(v)
    Pauli_string = list(Pauli_string)
    sampling_op_full = {}
    for _s in Pauli_string:
        sampling_op_full.setdefault(len(_s) - _s.count('I'), []).append(_s)

    # found sampling op
    op_size = list(sampling_op_full)
    sampling_op = {_op_size: [] for _op_size in op_size}
    existed_sampling_op = []
    for _op_size in sorted(op_size, reverse=True):
        for _sampling_op in sampling_op_full[_op_size]:
            regex = get_regex(_sampling_op)
            found_sampling_op = False
            for _existed_sampling_op in existed_sampling_op:
                _match = regex.match(_existed_sampling_op)
                if _match is not None:
                    found_sampling_op = True
                    break
            if not found_sampling_op:
                sampling_op[_op_size].append(_sampling_op)
                existed_sampling_op.append(_sampling_op)

    # # found map from result to sampling op
    sampling_op_full_map = {}
    for _op_size in sorted(op_size, reverse=True):
        for _sampling_op in sampling_op_full[_op_size]:
            regex = get_regex(_sampling_op)
            for _existed_sampling_op in existed_sampling_op:
                _match = regex.match(_existed_sampling_op)
                if _match is not None:
                    sampling_op_full_map.setdefault(_sampling_op,
                                                    []).append(_match.group())
    with open(root.joinpath('temp/decomposed_current_matrix_info.pkl'),
              'wb') as f:
        pickle.dump(
            {
                'sampling_op_full_map': sampling_op_full_map,
                'sampling_op': existed_sampling_op
            }, f)


def load_sampling_op_info():
    with open(root.joinpath('temp/decomposed_current_matrix_info.pkl'),
              'rb') as f:
        result = pickle.load(f)
    return result


if not root.joinpath('temp/decomposed_current_matrix_info.pkl').exists():
    dump_sampling_op_info()
SAMPLING_OP_INFO = load_sampling_op_info()


def prob_to_expect(probs: np.ndarray) -> float:
    '''probs: array'''
    qnum = int(math.log2(len(probs)))
    factors = reduce(np.outer, [np.array([1.0, -1.0])] * qnum).reshape(-1)
    result = np.sum(factors * probs)
    return result


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


def process_sampling_result(probs: dict, collect_expectation=False):
    N = 2**5
    decomposed_current_matrix, sampling_op_info = DECOMPOSED_CURRENT_MATRIX, SAMPLING_OP_INFO
    sampling_op_full_map = sampling_op_info['sampling_op_full_map']
    rho0 = probs['ZZZZZZZZZZ'].reshape(32, 32)
    current_x = np.zeros((N, N), dtype=np.complex128)
    current_y = np.zeros((N, N), dtype=np.complex128)
    _expectation_cache = {}
    for _Pauli_string, _target_ops in sampling_op_full_map.items():
        _target_op = _target_ops[0]  # we choose the first one
        remain_q_idxs = tuple(
            [idx for idx, _op in enumerate(_Pauli_string) if _op != 'I'])
        _probs = probs[_target_op]
        _expectation_cache[_Pauli_string] = prob_to_expect(
            ptrace(_probs, remain_q_idxs))

    for x in tqdm.tqdm(range(N), desc='processing sampling result'):
        for y in range(N):
            _jx = decomposed_current_matrix['jx'][(x, y)]
            _jy = decomposed_current_matrix['jy'][(x, y)]
            # calculate <jx>
            _jx_expectation = 0
            for _Pauli_string, _coe in _jx.items():
                if _Pauli_string == 'I' * len(_Pauli_string):
                    _jx_expectation += _coe
                else:
                    _jx_expectation += _coe * _expectation_cache[_Pauli_string]
            # calculate <jy>
            _jy_expectation = 0
            for _Pauli_string, _coe in _jy.items():
                if _Pauli_string == 'I' * len(_Pauli_string):
                    _jy_expectation += _coe
                else:
                    _jy_expectation += _coe * _expectation_cache[_Pauli_string]
            current_x[y][x] = _jx_expectation
            current_y[y][x] = _jy_expectation
    current_x = current_x.real
    current_y = current_y.real
    current = current_x + 1j * current_y

    if collect_expectation:
        return rho0, current, _expectation_cache
    else:
        return rho0, current
