from . import *
from math import pi


def build_circuit_deviation_of_Jy() -> simulator.Circuit:
    with open(root.joinpath('raw_circuit/diverging_flow_t=0.qasm'), 'r') as f:
        qasm = f.read()
    qasm = qasm.split(';\n')

    seed = 0
    print("random seed:", seed)
    np.random.seed(seed)
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
            circuit.qiskit_u(q_idx, 0.09 * np.random.rand() - 0.045,
                             0.09 * np.random.rand() - 0.045,
                             0.09 * np.random.rand() - 0.045)
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


def deviation_of_Jy_simulation(save=False):
    N = 2**5
    sampling_op = SAMPLING_OP_INFO['sampling_op']
    c0 = build_circuit_deviation_of_Jy()
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
        with pd.ExcelWriter(root.joinpath('temp/deviation_of_Jy.xlsx')) as ew:
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

    fig = plt.figure(figsize=[7, 7])
    ax = fig.add_subplot(2, 2, 1)
    im = ax.imshow(rho0, origin='lower', cmap='Blues', vmin=0, vmax=0.004)
    plt.colorbar(im, ax=ax, location='bottom', shrink=0.7)
    _title = 'density'
    ax.set_title(_title)
    set_ax(ax)

    ax = fig.add_subplot(2, 2, 2)
    im = ax.imshow(np.abs(current),
                   cmap='Greens',
                   origin='lower',
                   vmin=0,
                   vmax=0.004)
    strm = ax.streamplot(xs,
                         ys,
                         current.real,
                         current.imag,
                         density=1,
                         color=np.abs(current),
                         cmap='Greys',
                         linewidth=0.5)
    strm.lines.set_clim(0, 0.0025)
    plt.colorbar(im, ax=ax, location='bottom', shrink=0.7)
    _title = 'momentum'
    ax.set_title(_title)
    set_ax(ax)

    ax = fig.add_subplot(2, 5, (7, 9))
    exp_data_path = root.parent.parent.joinpath(
        "data_availability/Fig2/Exp_t=0.xlsx")
    exp_data = pd.read_excel(exp_data_path, sheet_name=None, index_col=0)
    y = np.array(exp_data['y'])[:, 0]
    J_y = np.array(exp_data['momentum_y'])
    plt.plot(y,
             np.mean(J_y, axis=1),
             marker='^',
             c='#007f00',
             label='exp',
             markerfacecolor='None')
    plt.plot(y,
             np.mean(current.imag, axis=1),
             marker='o',
             c='royalblue',
             label='sim',
             markerfacecolor='None')
    plt.legend()
    xticklabels = [r'-$\pi$', r'-$\pi/2$', 0, r'$\pi/2$', r'$\pi$']
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], xticklabels)
    plt.xlabel('y')
    plt.ylabel(r'$\langle J_y \rangle_x$')
