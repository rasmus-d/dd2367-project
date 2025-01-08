# Two Quantum computer simulators


## Project structure

### requirements.txt

Contains the project requirements; third-party libraries.

### src - quantum simulator library

Cotains all code and modules that make up the core of the two simulators.
Files prefixed with ```general_``` relates to the density matrix simulator using Choi matrix representation for the Channels.
The other files relates to the simulator using sparse vector representation.

### app - quantum simulator app/driver

Application that the user interfaces with to perform quantum simulations.

### run.sh

wrapper script executing the quantum simulator application.

