'''
If we want to add some tests later.
Probably a good idea but might just eat valuable time
'''
import unittest

from src import * 

def showplot(qc):
    qc.circplot()
    plt.savefig('test.png')

class TestQuantumSimulator(unittest.TestCase):

    def test_hadamard(self):
        qc = QuantumSimulator(1)
        qc.add(H(0))
        qc.run()
        
        chance0 = qc.state.dict[0]
        chance1 = qc.state.dict[0]
        self.assertAlmostEqual(chance0 * chance0, 0.5)
        self.assertAlmostEqual(chance1 * chance1, 0.5)
    def test_measure_hadamard(self):
        test = 10000
        count = 0 
        for _ in range(test):
            qc = QuantumSimulator(1)
            qc.add(H(0))
            qc.add(Measurement(0))
            if qc.run()[0][0]:
                count += 1
        percentage = count / test 
        self.assertGreater(percentage, 0.48)
        self.assertLess(percentage, 0.52)
    
    def test_X(self):
        qc = QuantumSimulator(1, State({0: 1}))
        qc.add(X(0))
        qc.run()
        self.assertEqual(qc.state.dict[1], 1)

        qc = QuantumSimulator(1, State({1: 1}))
        qc.add(X(0))
        qc.run()
        self.assertEqual(qc.state.dict[0], 1)

    def test_Y(self):
        qc = QuantumSimulator(1, State({0: 1}))
        qc.add(Y(0))
        qc.run()
        self.assertEqual(qc.state.dict[1], 1j)

        qc = QuantumSimulator(1, State({1: 1}))
        qc.add(Y(0))
        qc.run()
        self.assertEqual(qc.state.dict[0], -1j)
    
    def test_Z(self):
        qc = QuantumSimulator(1, State({0: 1}))
        qc.add(Z(0))
        qc.run()
        self.assertEqual(qc.state.dict[0], 1)

        qc = QuantumSimulator(1, State({1: 1}))
        qc.add(Z(0))
        qc.run()
        self.assertEqual(qc.state.dict[1], -1)
    
    def test_S(self):
        qc = QuantumSimulator(1, State({0: 1}))
        qc.add(S(0))
        qc.run()
        print(qc.state.dict)
        showplot(qc)

if __name__ == '__main__':
    unittest.main()