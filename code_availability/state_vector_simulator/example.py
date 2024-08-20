from . import simulator

def GHZ_test(q_num=10):
    print('-' * 10)
    print(f'**  {q_num}q GHZ state **')
    print('-' * 10)
    c = simulator.Circuit()
    c.plus_gate(0, 'H')
    for i in range(1, q_num):
        c.CNOT([0, i])
    print('counts:')
    print(c.state_vector().counts())
