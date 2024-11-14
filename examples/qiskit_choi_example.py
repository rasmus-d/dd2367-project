import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Choi
from qiskit_aer.primitives import SamplerV2
from qiskit.visualization import plot_histogram

circ = QuantumCircuit(1,1)

#circ.initialize([1/np.sqrt(2), 1/np.sqrt(2)], 0)
circ.initialize([0, 1], 0)

# Had-gate as unitary matrix.
uni_had = [[1/np.sqrt(2), 1/np.sqrt(2)],
           [1/np.sqrt(2), -1/np.sqrt(2)]]
uni_gate = UnitaryGate(uni_had)

# Choi representation of a Had-gate. The size is (2*2)x(2*2) 
# since we have 2 input states and two output states.
# See definition:
# https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/quantum-channels#definition
# In our case the channel \phi is \phi(p) = HpH\dagger
# according to formula (1) at
# https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/quantum-channels#unitary-operations-as-channels
choi_had = [[0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5, 0.5]]

# Example of non-unitary channel
#choi_completely_dephasing = [[1, 0, 0, 0],
#                             [0, 0, 0, 0],
#                             [0, 0, 0, 0],
#                             [0, 0, 0, 1]]

choi_channel = Choi(choi_had, 2, 2)

# You can compare applying thise by uncommenting,
# it should be the same for all initializations.
#circ.append(uni_gate, [0])
circ.append(choi_channel, [0])

circ.measure(0, 0)

# We can see on the drawed circuit that it in this case is simplified to a unitary gate.
# This is not the case if we use for example the choi_completely_dephasing matrix above
circ.draw("mpl")

# Nothing new here
sampler = SamplerV2()
job = sampler.run([circ], shots=100)
result = job.result()[0]
counts = result.data.c.get_counts()
plot_histogram(counts)

plt.show()