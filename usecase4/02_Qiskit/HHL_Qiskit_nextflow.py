import numpy as np
import scipy
import scipy.linalg
import qiskit
import qiskit.circuit
import qiskit_aer
import math
import sys
import io
import ast

import matplotlib.pyplot as plt
from typing import Tuple

import time
import pathlib
import json


class HHL_Qiskit:
    def __init__(self, A:np.ndarray|None=None, b:np.ndarray|None=None, c_reg_size:int|None=None, t_hamiltonian:float|None=None):
        self.A = A
        self.b = b
        self.c_reg_size = c_reg_size
        self.b_reg_size = int(np.log2(len(b))) if b is not None else None
        self.t_hamiltonian = t_hamiltonian
        self.circuit = None

    
    def _verify_hamiltonian_gate(self, A:np.ndarray, U:qiskit.circuit.library.HamiltonianGate) -> None:
        """Verify that the matrix exponential was computed correctly. This seems to be not very stable due to scipy.linalg.expm
        https://github.com/scipy/scipy/issues?q=is%3Aissue%20state%3Aopen%20expm
        """

        w, vr = scipy.linalg.eig(A, right=True)
        _AA = vr @ np.diag(w) @ vr.T
        is_eig_ok = np.allclose(A, _AA) #"eigenvalue decomposition deviates from A"

        sp_expm_A = scipy.linalg.expm(A)
        exp_diagonalized_A = vr @ np.diag(np.exp(w)) @ vr.T
        is_expm_ok = np.allclose(sp_expm_A, exp_diagonalized_A) #"expm deviates"

        is_hamiltoniangate_ok = np.allclose(sp_expm_A, U.to_matrix()) #"HamiltonianGate deviates"

        if not is_eig_ok:
            print("Eigenvalue decomposition is off")
            print("Eigenvalues:\n", w)
            print("Right eigenvectors:\n", vr)
            print("A:\n", A)
            print("vr @ diag(w) @ vr.T\n", _AA)
        if not is_expm_ok:
            print("Exponential of A scipy:\n", sp_expm_A)
            print("Exponential of A diagonalized:\n", exp_diagonalized_A)
        if not is_hamiltoniangate_ok:
            print("HamiltonianGate:\n", U.to_matrix())
            print("Exponential of A scipy:\n", sp_expm_A)
            print("Exponential of A diagonalized:\n", exp_diagonalized_A)


        assert is_eig_ok and is_expm_ok and is_hamiltoniangate_ok , "HamiltonianGate is off, see above"

    def construct_quantum_circuit(self, r:None=None,t_=None,c_reg_size=None,b_reg_size=None,A_herm=None,b_herm=None, include_measurements:bool=True):
        t_ = self.t_hamiltonian if t_ is None else t_
        c_reg_size = self.c_reg_size if c_reg_size is None else c_reg_size
        b_reg_size = self.b_reg_size if b_reg_size is None else b_reg_size
        A_herm = self.A_herm if A_herm is None else A_herm
        b_herm = self.b_herm_normalized if b_herm is None else b_herm
        
        
        
        #=========================== Quantum Circuit ========================================#
        # Qubits
        anc = qiskit.QuantumRegister(1,'anc')
        c_reg = qiskit.QuantumRegister(c_reg_size, 'c_reg')
        b_reg = qiskit.QuantumRegister(b_reg_size, 'b_reg')
        # Classical Bits
        if include_measurements:
            #cbit_anc = ClassicalRegister(1,'ancilla_cbit')
            cbit_reg = qiskit.ClassicalRegister(b_reg_size+c_reg_size+1, 'cregb')
            # Initialize circuit
            #circuit = qiskit.QuantumCircuit(anc,c_reg,b_reg,cbit_anc,cbit_reg)
            circuit = qiskit.QuantumCircuit(anc,c_reg,b_reg,cbit_reg)
        else:
            #circuit = qiskit.QuantumCircuit(anc,c_reg,b_reg)
            circuit = qiskit.QuantumCircuit(anc,c_reg,b_reg)
        # Initialize state b
        #init = qiskit.circuit.Initialize(list(b_herm.flatten()))
        #circuit.append(init,b_reg)
        if self.b_herm_normalized.size==2 and np.allclose(self.b_herm.flatten(), [0,1]):
            circuit.x(b_reg[0])
        elif self.b_herm_normalized.size==2 and np.allclose(self.b_herm.flatten(), [1,0]):
            pass
        elif np.allclose(self.b_herm_normalized.flatten(), 1/np.linalg.norm(self.b_herm)*np.ones_like(self.b_herm.flatten())):
            circuit.h(b_reg)
        else:
            assert False, f"initialization for b_herm_normalized not implemented, b_herm_normalized: {self.b_herm_normalized.flatten()}"
        circuit.barrier(label='init_b')
        #circuit.draw()
        # Apply H-gate on quantum register a
        circuit.h(c_reg)
        circuit.barrier(label='init_c')
        # Apply controlled Hamiltonian operators on quantum register b
        for i in range(c_reg_size):
            #sim_time = t_/(2**(c_reg_size-i))
            sim_time = t_*(2**(i))
            U = qiskit.circuit.library.HamiltonianGate( - A_herm, sim_time, label='H'+str(i)) #HamiltonianGate automatically includes -1j
            self._verify_hamiltonian_gate(1j*A_herm*sim_time, U)
            #U = U.conjugate()
            #print('i, U:', i, U.to_matrix())
            #print('U', scipy.linalg.expm(- 1j * sim_time * A_herm))
            #print('U diff:', U.to_matrix()-scipy.linalg.expm(- 1j * sim_time * A_herm))
            #assert np.allclose(U.to_matrix(), scipy.linalg.expm(- 1j * sim_time * A_herm))

            G = U.control(1)
            qubit = [i+1]+[c_reg_size+j+1 for j in range(b_reg_size)]
            circuit.append(G,qubit)
        circuit.barrier(label='cHamil_b')
        # Apply inverse Quantum Fourier Transform
        iqft = qiskit.circuit.library.QFT(c_reg_size, approximation_degree=0, do_swaps=True, inverse=True, name='IQFT')
        circuit.append(iqft, c_reg)
        circuit.barrier(label='iqft')

        # Swap Qubits in quantum register A
        G = qiskit.circuit.library.SwapGate()
        #circuit.append(G,[c_reg[1],c_reg[c_reg_size-1]])
        for i in range(c_reg_size):
            a = (c_reg_size - 1) - i
            b = i
            if not (a == b) and not (a < b):
                circuit.append(G,[c_reg[a],c_reg[b]])

        circuit.barrier(label='swap')
        #Conditioned Rotation of Ancilla#
        for i in range(c_reg_size):
            theta = 2*math.asin(1/(2**i))
            U = qiskit.circuit.library.RYGate(theta).control(1)
            circuit.append(U,[i+1,0])
        circuit.barrier(label='rot')
        
        #=============================== Uncompute the Circuit =========================#
        # Swap qubits in quantum register A
        G = qiskit.circuit.library.SwapGate()
        #circuit.append(G,[c_reg[1], c_reg[c_reg_size-1]])
        for i in range(c_reg_size, -1, -1):
            a = (c_reg_size - 1) - i
            b = i
            if not (a == b) and not (a < b):
                circuit.append(G,[c_reg[a],c_reg[b]])


        circuit.barrier(label='swap')
        # Apply Quantum Fourier Transform
        qft = qiskit.circuit.library.QFT(c_reg_size, approximation_degree=0, do_swaps=True, inverse=False, name='QFT')
        circuit.append(qft, c_reg)
        circuit.barrier(label='qft')
        # Apply inverse controlled Hamiltonian Operators
        for i in range(c_reg_size-1,-1,-1):
            #time = t_/(2**(c_reg_size-i))
            time = t_*(2**(i))
            U = qiskit.circuit.library.HamiltonianGate( - A_herm, time, label='H'+str(i)) #HamiltonianGate automatically includes -1j
            self._verify_hamiltonian_gate(1j*A_herm*time, U)
            #U = U.conjugate()
            #print('i, U:', i, U.to_matrix())
            G = U.control(1)
            G = G.inverse()
            qubit = [i+1]+[c_reg_size+j+1 for j in range(b_reg_size-1,-1,-1)]
            circuit.append(G,qubit)
        # Apply H Gate on Quantum Register A
        circuit.barrier(label='cHamil_b')
        circuit.h(c_reg)
        circuit.barrier(label='init_c')
        # Measure the qubits
        #circuit.measure(anc, cbit_anc)
        if include_measurements:
            circuit.measure(anc, cbit_reg[0])
            circuit.measure(c_reg, cbit_reg[1:1+c_reg_size])
            circuit.measure(b_reg, cbit_reg[1+c_reg_size:1+c_reg_size+b_reg_size])
        
        self.circuit = circuit
        
        # Return constructed circuit
        return circuit



    def _hermitianize(self, mat:np.matrix) -> np.ndarray:
        herm = np.zeros((2*mat.shape[0], 2*mat.shape[1]))
        mat_conj = mat.getH()
        for i in range(herm.shape[0]):
            for j in range(herm.shape[0]):
                if i < mat.shape[0] and j >= mat.shape[0]:
                    herm[i,j] = mat[i,j-mat.shape[0]]
                elif i >= mat.shape[0] and j < mat.shape[0]:
                    herm[i,j] = mat_conj[i-mat.shape[0],j]
                else:
                    pass
        return herm

    def hermitianize_system(self, A=None,b=None) -> Tuple[np.ndarray,np.ndarray]:

        if A is None:
            A = self.A
        if b is None:
            b = self.b

        if A.shape[0] != A.shape[1]:
            print('Not a Square Matrix')
            sys.exit(0)
        if int(np.log2(A.shape[0])) == float(np.log2(A.shape[0])):
            bs = int(np.log2(A.shape[0]))
        else:
            print('Matrix size is not a power of 2')
            sys.exit(0)
        A_new = np.asmatrix(A)
        b_new = b
        if not scipy.linalg.ishermitian(A_new):
            print('Matrix A is not hermitian, system will be Hermitianized as [[A,0],[0,A^dagger]]')
            print('Matrix A:\n', A)
            # Hermitianize the input matrix
            A_hermitian = self._hermitianize(A_new)
            # Adjust the right side vector with additional zeros
            b_hermitian = np.zeros(2*b_new.shape[0])
            b_hermitian[:b_new.shape[0]] = b_new
            # Change hermitian matrix data type from matrix to array
            A_hermitian = np.asarray(A_hermitian)
        else:
            print('Matrix A is hermitian, system is not changed')
            # Do nothing, the matrix is hermitian
            A_hermitian = np.asarray(A_new)
            b_hermitian = b_new
        self.A_herm = A_hermitian
        self.b_herm = b_hermitian
        b_herm_normalized = b_hermitian/np.linalg.norm(b_hermitian)
        self.b_herm_normalized = b_herm_normalized

        self.b_reg_size = int(np.log2(len(b_herm_normalized)))

        self.t_hamiltonian = np.pi/max(scipy.linalg.eigvals(A_hermitian).real) if self.t_hamiltonian is None else self.t_hamiltonian
        return (A_hermitian, b_hermitian, b_herm_normalized)

    def compute_classical_solutions(self):
        assert self.A is not None and self.A_herm is not None, "Matrix A and/or A_herm is not set"
        assert self.b is not None and self.b_herm is not None, "Vector b and/or b_herm is not set"

        sol_classical = np.linalg.solve(self.A, self.b).flatten()
        sol_classical_herm = np.linalg.solve(self.A_herm, self.b_herm).flatten()
        sol_classical_herm_normalized = np.linalg.solve(self.A_herm, self.b_herm_normalized).flatten()
        
        self.sol_classical = sol_classical
        self.sol_classical_herm = sol_classical_herm
        self.sol_classical_herm_normalized = sol_classical_herm_normalized
        return (sol_classical, sol_classical_herm, sol_classical_herm_normalized)

def read_config_from_json(file:str|pathlib.Path)->dict:
    """
    Read configuration from a JSON file.
    """
    required_keys = ['A', 'b', 'c_reg_size', 't_hamiltonian', 'shots']
    items_arrays = ['A', 'b', 'samples_discrete']
    keys_not_found_in_json = []
    with open(file, 'r') as f:
        dict_config = json.load(f)
    for key in required_keys:
        if key not in dict_config and key not in dict_config['col_names']:
            keys_not_found_in_json.append(key)
    if len(keys_not_found_in_json) > 0:
        raise ValueError(f"Required keys ({keys_not_found_in_json}) not found in JSON file")
    for key in items_arrays:
        dict_config[key] = np.load(io.BytesIO(ast.literal_eval(dict_config[key])))[key]
    return dict_config

if __name__ == "__main__":

    # Read configuration from JSON file
    config_file = pathlib.Path().cwd().glob('dict_params_run_*.json')
    config_file = [x for x in config_file]
    assert len(config_file) == 1, f"Found {len(config_file)} params files, expected 1"
    config_file = config_file[0]
    dict_config = read_config_from_json(config_file)

    print('dict_config', dict_config)

    is_print_figs_circuits = False
    is_print_figs_result = False

    system_size = dict_config['A'].shape[0]
    #device = dict_config['device'] # 'GPU' or 'CPU'
    device = 'GPU'
    shots = dict_config['shots']
    t_hamiltonian = dict_config['samples_discrete'][dict_config['col_names'].index('t_hamiltonian')]
    c_reg_size = int(dict_config['samples_discrete'][dict_config['col_names'].index('c_reg_size')])
    A = dict_config['A']
    b = dict_config['b']

    hhl_instance = HHL_Qiskit(A=A, b=b, c_reg_size=c_reg_size, t_hamiltonian=t_hamiltonian)
    A_herm, b_herm, b_herm_normalized = hhl_instance.hermitianize_system()

    def dont_need_just_here_for_collapsable_referene():
        # Define params
        system_size = 16 # see cases below
        device = 'CPU' # 'GPU' or 'CPU'
        is_print_figs_circuits = False
        is_print_figs_result = True

        if system_size == 4:
            c_reg_size = 5 # playaround determined this to work somewhat well

            hhl_instance = HHL_Qiskit(c_reg_size=c_reg_size)

            A,b,alpha = hhl_instance.init_Ab_poisson_first_order_FD(system_size=system_size)
            A_herm, b_herm, b_herm_normalized = hhl_instance.hermitianize_system()

        elif system_size > 4:
            c_reg_size = system_size

            hhl_instance = HHL_Qiskit(c_reg_size=c_reg_size)

            A,b,alpha = hhl_instance.init_Ab_poisson_first_order_FD(system_size=system_size)
            A_herm, b_herm, b_herm_normalized = hhl_instance.hermitianize_system()

        elif system_size == 2:
            c_reg_size = 4
            A_herm = np.array([[1, -1/3], [-1/3, 1]])
            b_herm = np.array([[0], [1]])
            hhl_instance = HHL_Qiskit(A=A_herm, b=b_herm, c_reg_size=c_reg_size)
            A_herm, b_herm, b_herm_normalized = hhl_instance.hermitianize_system()


    sol_classical_herm = hhl_instance.compute_classical_solutions()[2]
    

    # Construct the quantum circuit
    qc = hhl_instance.construct_quantum_circuit()
    if is_print_figs_circuits:
        qc.draw('mpl')
        plt.savefig(f'HHL_circuit_Qiskit_{system_size}x{system_size}.png')

    #print('Available devices to run simulator on:', qiskit_aer.AerSimulator(method='statevector').available_devices())
    sim = qiskit_aer.AerSimulator(method='statevector', device=device)
    qc_transpiled = qiskit.transpile(circuits=qc,backend=sim, optimization_level=0)#, basis_gates=instructions)
    
    tic = time.time()
    job = sim.run(qc_transpiled, shots=shots)
    results = job.result()
    counts = results.get_counts()
    toc_compute_samples = time.time()
    #print('Time to compute samples:', toc_compute_samples - tic)

    counts_array = np.zeros((len(counts), 2), dtype=np.int64)
    qsol = np.zeros(len(b_herm))
    for i, key in enumerate(counts):
        counts_array[i] = [int(key), counts[key]]
        counts[key] *= (1/shots)
        counts[key] = np.sqrt(counts[key])
    for key in counts:
        if key[-1] == '1':
            pos = int(key[0:hhl_instance.b_reg_size],2)
            qsol[pos] +=counts[key]
    
    outfile = 'counts_array_id_{}.npz'.format(int(dict_config['samples_discrete'][dict_config['col_names'].index('id')]))
    with open(outfile, 'wb') as f:
        np.savez_compressed(f, counts_array=counts_array)
    
    #print(np.load('counts_array.npz')['counts_array'])
    #print('counts:', counts)
    #print('Quantum solution, ratio elem 0/1:', qsol, qsol[0]/qsol[1])
    #print('Quantum solution normalized, norm:', qsol/np.linalg.norm(qsol), np.linalg.norm(qsol))
    #print('Classical solution, ratio elem 0/1:', sol_classical_herm, sol_classical_herm[0]/sol_classical_herm[1])
    #print('Classical solution normalized, norm:', sol_classical_herm/np.linalg.norm(sol_classical_herm), np.linalg.norm(sol_classical_herm))

    
    if is_print_figs_circuits:
        qc_transpiled.draw('mpl')
        plt.savefig(f'HHL_circuit_transpiled_Qiskit_{system_size}x{system_size}.png')

    if is_print_figs_result:
        fig, ax = plt.subplots()
        ax.plot(qsol)
        fig.savefig(f'HHL_result_Qiskit_{system_size}x{system_size}.png')


