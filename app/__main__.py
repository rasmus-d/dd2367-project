from src import *

'''
TODO: Density matrix qft with noise. Compare with qiskit.
'''
def state_vector_qft(n:int) -> List[Operator | Measurement]:
    queue = []
    for q in range(n-1,-1,-1):
        queue.append(H(q))
        for i in range(q-1,-1,-1):
            queue.append(P(q, -pi/(2*(q-i)), control=[i]))
    for i in range(int(n/2)):
        queue.append(SWAP(i,n-i-1))
    return queue

def density_matrix_qft(n:int) -> List[Channel | GeneralMeasurement]:
    queue = []
    for q in range(n-1,-1,-1):
        queue.append(QChannel(q, HChannel()))
        for i in range(q-1,-1,-1):
            queue.append(QControlledU(i, q, PChannel(-pi/(2*(q-i))))) #TODO: Implement controlled P or controlled gates in general. Control always < target!
    for i in range(int(n/2)):
        queue.append(QChannel(i, SwapChannel(n-i-1 - i))) #TODO: Implement swap of two qubits with arbitrary distance
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
    sim.add(state_vector_qft(4))
    res = sim.run()
    sim.circplot()
    print(sim.state)
    print(res)

def density_matrix_example():
    size = 4
    sim = MatrixSimulator(size)
    for i in range(size):
        sim.add(QChannel(i, HChannel()))
    for i in range(size-1):
        sim.add(QChannel(i, PChannel(pi/2**(size-2-i))))

    sim.add(density_matrix_qft(size))
    sim.add(StandardBasisMeasurement(size))
    res = sim.run()
    #If measurement:
    print("Density matrix res: \n", res)
    #If no measurement:
    #print("res.density_matrix: \n", res.density_matrix)

def main():
    density_matrix_example()
if __name__ == '__main__':
    main()

