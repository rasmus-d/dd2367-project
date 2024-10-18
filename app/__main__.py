from src import *


def main():
    sim = QuantumSimulator(42)
    sim.add(H(31,[]))
    res = sim.run()
    print(res)


if __name__ == '__main__':
    main()

