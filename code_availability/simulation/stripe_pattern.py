from . import *
from math import pi


def build_circuit_stripe_pattern(error_q_idx=6) -> simulator.Circuit:
    with open(root.joinpath('raw_circuit/diverging_flow_t=pi_over_2.qasm'),
              'r') as f:
        qasm = f.read()
    qasm = qasm.split(';\n')
    circuit = simulator.Circuit()
    for gate_info in qasm[3:-1]:
        _gate, _q = gate_info.split(' ')
        _q = _q.split(',')
        if len(_q) == 1:
            q_idx = int(_q[0][2:-1])
            if _gate.startswith('u'):
                theta, phi, angle_lambda = [
                    eval(angle)
                    for angle in re.findall('u\((.*),(.*),(.*)\)', _gate)[0]
                ]
                circuit.qiskit_u(q_idx, theta, phi, angle_lambda)
            else:
                raise NotImplementedError(_gate)
            if q_idx == error_q_idx:
                circuit.Rx(error_q_idx, 0.025)
        elif len(_q) == 2:
            q0_idx, q1_idx = _q
            q0_idx, q1_idx = int(q0_idx[2:-1]), int(q1_idx[2:-1])
            if _gate == 'cz':
                circuit.CZ([q0_idx, q1_idx])
            else:
                raise NotImplementedError(_gate)
        else:
            raise Exception(f'unknown gate {_gate} for {gate_info}')

    return circuit


def stripe_pattern_simulation(error_q_idx=6, save=False):
    N = 2**5
    sampling_op_info = load_sampling_op_info()
    sampling_op = sampling_op_info['sampling_op']
    c0 = build_circuit_stripe_pattern(error_q_idx)
    probs = {'ZZZZZZZZZZ': c0.state_vector().probs(list(range(10)))}
    for _sampling_op in sampling_op:
        _c = c0.copy()
        for _idx, _op in enumerate(_sampling_op):
            if _op == 'X':
                _c.plus_gate(_idx, '-Y/2')
            elif _op == 'Y':
                _c.plus_gate(_idx, 'X/2')
        probs[_sampling_op] = _c.state_vector().probs(list(range(10)))
    rho0, current, expectation = process_sampling_result(
        probs, collect_expectation=True)

    if save:
        _x = np.arange(-np.pi, np.pi, 2 * np.pi / 2**5)
        _y = np.arange(-np.pi, np.pi, 2 * np.pi / 2**5)
        _xs, _ys = np.meshgrid(_x, _y)
        with pd.ExcelWriter(
                root.joinpath(
                    f"temp/stripe_pattern_q{error_q_idx}_error.xlsx")) as ew:
            pd.DataFrame(_xs).to_excel(ew, sheet_name='x')
            pd.DataFrame(_ys).to_excel(ew, sheet_name='y')
            pd.DataFrame(rho0).to_excel(ew, sheet_name='density')
            pd.DataFrame(current.real).to_excel(ew, sheet_name='momentum_x')
            pd.DataFrame(current.imag).to_excel(ew, sheet_name='momentum_y')
            pd.DataFrame({
                'expectation': expectation
            }).to_excel(ew, sheet_name='expectation')

    x = np.linspace(0, 2**5 - 1, 2**5)
    y = np.linspace(0, 2**5 - 1, 2**5)
    xticks = np.linspace(0, 2**5 - 1, 5)
    yticks = np.linspace(0, 2**5 - 1, 5)
    xticklabels = [r'-$\pi$', r'-$\pi/2$', 0, r'$\pi/2$', r'$\pi$']
    yticklabels = [r'-$\pi$', r'-$\pi/2$', 0, r'$\pi/2$', r'$\pi$']
    xs, ys = np.meshgrid(x, y)

    def set_ax(ax):
        ax.set_xlim(0, 2**5 - 1)
        ax.set_ylim(0, 2**5 - 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig = plt.figure(figsize=[7, 4])
    ax = fig.add_subplot(1, 2, 1)
    # im = ax.imshow(rho0, origin='lower', cmap='Blues', vmin=0, vmax=0.0025)
    im = ax.imshow(rho0, origin='lower', cmap='Blues', vmin=0, vmax=0.004)
    plt.colorbar(im, ax=ax, location='bottom')
    _title = 'density'
    ax.set_title(_title)
    set_ax(ax)

    ax = fig.add_subplot(1, 2, 2)
    # im = ax.imshow(np.abs(current),
    #                cmap='Blues',
    #                origin='lower',
    #                vmin=0,
    #                vmax=0.001)
    im = ax.imshow(np.abs(current),
                   cmap='Greens',
                   origin='lower',
                   vmin=0,
                   vmax=0.005)
    strm = ax.streamplot(xs,
                         ys,
                         current.real,
                         current.imag,
                         density=1,
                         color=np.abs(current),
                         cmap='Greys',
                         linewidth=0.5)
    # strm.lines.set_clim(0, 0.0005)
    strm.lines.set_clim(0, 0.0025)
    plt.colorbar(im, ax=ax, location='bottom')
    _title = 'momentum'
    ax.set_title(_title)
    set_ax(ax)

    plt.tight_layout()
