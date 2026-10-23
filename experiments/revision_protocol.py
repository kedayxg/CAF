"""Fixed community capacities for the arXiv v2 rerun protocol.

Classical capacity 100 is the user-selected resource scenario. Quantum
capacity 16 matches the available statevector simulation scale. Neither
number is a measured hardware limit or a capacity chosen by a gain sweep.
"""

CLASSICAL_CAPACITY = 100
QUANTUM_CAPACITY = 16
PROTOCOL_NAME = "igraph-newman-induced-capacity-v2"
