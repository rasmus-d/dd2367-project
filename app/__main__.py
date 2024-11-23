from src import *

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

def density_matrix_example():

    state = GeneralState(initial_matrix = np.array([[0.5,0.5,0.5],
                                                    [0.5,0.5+0.2j,0.7],
                                                    [0.3,0.5,0.77]]))
    print("state:\n", state.density_matrix)
    dephasing = CompletelyDephasingChannel(3)
    print("dephasing:\n", dephasing.choi_matrix)
    state2 = dephasing.apply(state)
    print("dephased state:\n", state2.density_matrix)

def main():
    density_matrix_example()

if __name__ == '__main__':
    main()

