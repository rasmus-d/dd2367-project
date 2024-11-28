from src import *

'''
TODO: Density matrix qft with noise. Compare with qiskit.
'''
def qft(n:int) -> List[Operator | Measurement]:
    queue = []
    for q in range(n-1,-1,-1):
        queue.append(H(q))
        for i in range(q-1,-1,-1):
            queue.append(P(q, -pi/(2*(q-i)), control=[i]))
    for i in range(int(n/2)):
        queue.append(SWAP(i,n-i-1))
    return queue

def state_vector_example() :
    sim = QuantumSimulator(4)
    sim.add(H(0))    
    sim.add(H(1))    
    sim.add(H(2))    
    sim.add(H(3))    
    sim.add(P(0,pi/4))
    sim.add(P(1,pi/2))
    sim.add(P(2,pi))
    sim.add(qft(4))
    res = sim.run()
    sim.circplot()
    print(sim.state)
    print(res)

def ex_dephase():
    bell_state = GeneralState(initial_matrix = np.array([[0.5,0,0,0.5],
                                                         [0,0,0,0],
                                                         [0,0,0,0],
                                                         [0.5,0,0,0.5]]))
    dephase = CompletelyDephasingChannel(4)
    state2 = dephase.apply(bell_state)
    print("state2: \n", state2.density_matrix)

def ex_reset():
    state = GeneralState(initial_matrix=np.array([[0,0],[0,1]]))
    reset = QubitResetChannel()
    state2 = reset.apply(state)
    print("state2: \n", state2.density_matrix)


def density_matrix_example():

    bell_state = GeneralState(initial_matrix = np.array([[0.5,0,0,0.5],
                                                         [0,0,0,0],
                                                         [0,0,0,0],
                                                         [0.5,0,0,0.5]]))

    sim = MatrixSimulator(num_qubits=2, initial_state=bell_state)

    '''
    We apply a channel to the second qubit.
    Observe: We specify how many states the preceeding and following system has by m and n.
    We have 0 qubits after the system where we apply this channel, which means that we have one
    classical state there since 2^0 = 1. Therefore, n=1
    '''
    qch = QChannel(pos=1, channel=CompletelyDepolarizingChannel())
    sim.add(qch)
    sim.add(StandardBasisMeasurement(2))

    probs = sim.run()
    print("probs:\n", probs)

def main():
    density_matrix_example()

if __name__ == '__main__':
    main()

