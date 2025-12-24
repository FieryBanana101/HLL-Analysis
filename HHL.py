import time
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import RYGate, UnitaryGate
from qiskit_aer import AerSimulator



def find_encoding_scheme(A, n):

    # Get the eigenvalue
    evals = np.linalg.eigvalsh(A)
    N = 2**n
    MAX_ITER = 25000

    # Iterate as many times as we can to find the correct 't' such that an integer encoding scheme is found, and unique between 1..2**(n-1)
    for i in range(1, MAX_ITER):
        t = i * np.pi / 4
        scaled = (N * evals * t) / (2 * np.pi)
        rounded = np.round(scaled).astype(int)
        
        if np.allclose(scaled, rounded) and np.all((rounded > 0) & (rounded < N)) and len(set(rounded)) == len(rounded):
            encoding = [format(v, f'0{n}b') for v in rounded]
            return t, encoding[0], encoding[1]

    return None, None, None



def HHL(A, b):
    
    # Defining the registers
    nc = 2
    clock = 2
    ancilla = 1
    qr_clock1 = QuantumRegister(1, 'clock_1')
    qr_clock2 = QuantumRegister(1, 'clock_2')
    qr_b = QuantumRegister(1, 'b')
    qr_ancilla = QuantumRegister(ancilla, 'ancilla')
    cl = ClassicalRegister(nc, 'cl')
    qc = QuantumCircuit(qr_ancilla, qr_clock1, qr_clock2, qr_b, cl)

    # Finding appropriate encoding scheme
    t, qubits1, qubits2 = find_encoding_scheme(A, nc)
    assert t, "No valid encoding scheme found!"

    # Constructing the controlled unitary gate
    U1 = expm(1j * A * t)
    U2 = expm(1j * A * 2 * t)
    gate_u1 = UnitaryGate(U1, label="exp(iAt)")
    gate_u2 = UnitaryGate(U2, label="exp(iA2t)")

    # Applying state preparation
    qc.x(qr_b)
    qc.barrier()

    # Applying the uniformly distributed superposition
    qc.h(qr_clock1)
    qc.h(qr_clock2)
    qc.barrier()

    # Applying the controlled unitary
    qc.append(gate_u1.control(1), [qr_clock1, qr_b[0]])
    qc.append(gate_u2.control(1), [qr_clock2, qr_b[0]])
    qc.barrier()

    # Applying IQFT
    qc.h(qr_clock2)
    qc.cp(-np.pi/2, qr_clock1, qr_clock2)
    qc.h(qr_clock1)
    qc.swap(qr_clock1, qr_clock2)
    qc.barrier()

    # Applying the controlled rotation on the ancilla
    qc.cry(np.pi, qr_clock1, qr_ancilla[0])
    qc.cry(np.pi/3, qr_clock2, qr_ancilla[0])
    qc.barrier()

    # Applying the QFT
    qc.swap(qr_clock1, qr_clock2)
    qc.h(qr_clock1)
    qc.cp(np.pi/2, qr_clock1, qr_clock2)
    qc.h(qr_clock2)
    qc.barrier()
    qc.measure(qr_ancilla, cl[ancilla:])
    qc.barrier()

    # Applying the inverse controlled unitary
    qc.append(gate_u2.inverse().control(1), [qr_clock2, qr_b[0]])
    qc.append(gate_u1.inverse().control(1), [qr_clock1, qr_b[0]])
    qc.barrier()

    # Collapse superposition, measure and return circuit with its encoding scheme
    qc.h(qr_clock1)
    qc.h(qr_clock2)
    qc.barrier()
    qc.measure(qr_b, cl[:ancilla])
    return qc, qubits1, qubits2



test_cases = [
    {"A": np.array([[1.5, 0.5], [0.5, 1.5]]), "b": np.array([1.0, 0.0])},
    {"A": np.array([[1.5, 0.5], [0.5, 1.5]]), "b": np.array([0.0, 1.0])},
    {"A": np.array([[1, -1/3], [-1/3, 1]]), "b": np.array([0.0, 1.0])},
    {"A": np.array([[2.0, 1.0], [1.0, 2.0]]), "b": np.array([1.0, 1.0])},
    {"A": np.array([[2.5, 0.5], [0.5, 2.5]]), "b": np.array([2.0, 1.0])},
    {"A": np.array([[3.0, 1.0], [1.0, 3.0]]), "b": np.array([0.0, 1.0])},
    {"A": np.array([[1.0, 0.5], [0.5, 1.0]]), "b": np.array([1.0, 2.0])},
    {"A": np.array([[1, -1/3], [-1/3, 1]]), "b": np.array([1.0, 2.0])},
    {"A": np.array([[2.0, 1.0], [1.0, 2.0]]), "b": np.array([3.0, 3.0])},
    {"A": np.array([[1.25, 0.25], [0.25, 1.25]]), "b": np.array([0.0, 2.0])},
]


if __name__ == "__main__":
    for idx, test in enumerate(test_cases):

        A_input, b_input = test["A"], test["b"]
        print(f"Test case {idx + 1}:\nA = {A_input}\nb = {b_input}\n", end='')

        # Classical algorithm
        t1 = time.perf_counter()
        gauss_jordan_result = np.linalg.solve(A_input, b_input)
        t2 = time.perf_counter()
        actual_ratio = gauss_jordan_result[0] ** 2 / gauss_jordan_result[1] ** 2
        print(f"Actual Ratio (Gauss-Jordan Elimination): {actual_ratio} ({t2 - t1} ms)")

        try:
            t1 = time.perf_counter_ns()

            # Construct the quantum circuit and encoding scheme
            hhl_qc, qubits1, qubits2 = HHL(A_input, b_input)

            # Run in Aer-simulator
            simulator = AerSimulator()
            new_circuit = transpile(hhl_qc, simulator)
            job = simulator.run(new_circuit, shots = 1000000)
            result = job.result()
            counts = result.get_counts()
            total_shots = sum(counts.values())
            probabilities = {state: count / total_shots for state, count in counts.items()}
            plot_distribution = (counts)

            # Get the approximate ratio
            prob_x1 = counts[qubits1]
            prob_x2 = counts[qubits2]
            approx_ratio = prob_x1**2 / prob_x2**2

            t2 = time.perf_counter_ns()
            print(f"Approximated Ratio (HHL Algorithm): {approx_ratio} ({t2 - t1} ms)\n\n")

        except AssertionError:
            print("Approximated Ratio (HHL Algorithm): Unable to find valid encoding scheme for this test case!\n\n")