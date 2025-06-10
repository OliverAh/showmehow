import numpy as np
import scipy
import scipy.linalg
import qiskit
import qiskit.qasm2
import qiskit_aer
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.joinpath('02_Qiskit').resolve()))
sys.path.append(str(pathlib.Path.cwd()))

import matplotlib.pyplot as plt
import pennylane as qml
import HHL_Qiskit_nextflow # contains functions to read configuration from JSON file, and to build the HHL circuit

import time

if __name__ == "__main__":

    # Read configuration from JSON file
    config_file = pathlib.Path().cwd().glob('dict_params_run_*.json')
    config_file = [x for x in config_file]
    assert len(config_file) == 1, f"Found {len(config_file)} params files, expected 1"
    config_file = config_file[0]
    dict_config = HHL_Qiskit_nextflow.read_config_from_json(config_file)

    print('dict_config', dict_config)

    is_print_figs_circuits = False
    is_print_figs_result = False

    system_size = dict_config['A'].shape[0]
    #device = dict_config['device'] # 'lightning.qubit' or 'default.qubit' or 'lightning.gpu'
    sample_or_state = 'sample' # 'state' or 'sample'
    device = 'lightning.qubit'
    shots = dict_config['shots']
    t_hamiltonian = dict_config['samples_discrete'][dict_config['col_names'].index('t_hamiltonian')]
    c_reg_size = int(dict_config['samples_discrete'][dict_config['col_names'].index('c_reg_size')])
    A = dict_config['A']
    b = dict_config['b']

    hhl_instance = HHL_Qiskit_nextflow.HHL_Qiskit(A=A, b=b, c_reg_size=c_reg_size, t_hamiltonian=t_hamiltonian)
    A_herm, b_herm, b_herm_normalized = hhl_instance.hermitianize_system()

    
    
    sol_classical_herm = hhl_instance.compute_classical_solutions()[2]

    
    qc = hhl_instance.construct_quantum_circuit(include_measurements=False)
    
    if is_print_figs_circuits:
        qc.draw('mpl')
        plt.savefig('HHL_circuit_Qiskit.png')
    
    sim = qiskit_aer.AerSimulator(method='statevector')
    #print(sim.available_devices())
    instructions = [so.name for so in sim.operations if isinstance(so, qiskit.circuit.Instruction)]
    #print('Instructions:', instructions)
    qc_transpiled = qiskit.transpile(circuits=qc ,backend=sim, optimization_level=0)#, basis_gates=instructions)
    if is_print_figs_circuits:
        qc_transpiled.draw('mpl')
        plt.savefig('HHL_circuit_Qiskit_transpiled.png')


    
    qc_qasm2 = qiskit.qasm2.dumps(qc_transpiled) # pennylane can only handle qasm2
    #qc_pennylane = qml.from_qiskit(qc)
    qc_pennylane_from_qasm2 = qml.from_qasm(qc_qasm2)
    

    if sample_or_state == 'state':
    
        dev = qml.device(device, wires=qc_transpiled.width())

        tic = time.time()
        @qml.qnode(dev)
        def circuit_state():
            qc_pennylane_from_qasm2()#wires=["a", "b", "c", "d"])
            return qml.state() # pennylane uses BIG endian ordering
        toc_compute_state = time.time()
        print('Time to compute state:', toc_compute_state - tic)
        bitstrings_states = [format(i, '0' + str(qc_transpiled.width()) + 'b')[::-1] for i in range(2**qc_transpiled.width())]
        #print(bitstrings_states)
        state = circuit_state()
        dict_state = {e[0]: e[1] for e in zip(bitstrings_states, state)}
        dict_probs = {e[0]: e[1] for e in zip(bitstrings_states, np.abs(state)**2)}
        if is_print_figs_circuits:
            fig, ax = qml.draw_mpl(circuit_state)()
    
    
    elif sample_or_state == 'sample':
        dev = qml.device(device, wires=qc_transpiled.width(), shots=shots)

        tic = time.time()
        @qml.qnode(dev)
        def circuit_sample():
            qc_pennylane_from_qasm2()
            #return qml.counts(all_outcomes=True) # pennylane uses BIG endian ordering
            return qml.counts() # pennylane uses BIG endian ordering
        samples = circuit_sample()
        toc_compute_samples = time.time()
        print('Time to compute samples:', toc_compute_samples - tic)
        #print(samples)
        #dict_probs = {e[0]: e[1] for e in samples.items()}
        if is_print_figs_circuits:
            fig, ax = qml.draw_mpl(circuit_sample)()
    
    #print(dict_probs)
    #print('Sum of probs:', np.sum(list(dict_probs.values())))
    
    if is_print_figs_circuits:
        fig.savefig('HHL_circuit_Pennylane.png')

    counts_array = np.zeros((len(samples), 2), dtype=np.int64)
    qsol = np.zeros(len(b_herm))
    for i, key in enumerate(samples.keys()):
        counts_array[i] = [int(key), samples[key]]
        if key[0] == '1':
            pos = int(key[-hhl_instance.b_reg_size:],2)
            #pos = int(key[0],2)
            #print(key, pos, dict_probs[key])
            qsol[pos] +=samples[key]
            #print('key:', key)
            #print('pos:', pos)
    #print(counts_array)
    #print(samples)
    outfile = 'counts_array_id_{}.npz'.format(int(dict_config['samples_discrete'][dict_config['col_names'].index('id')]))
    with open(outfile, 'wb') as f:
        np.savez_compressed(f, counts_array=counts_array)

    # print('qsol:', qsol, qsol[0]/qsol[1])
    # print('qsol normalized:', qsol/np.linalg.norm(qsol), np.linalg.norm(qsol))
    # print('Classical Solution:', sol_classical_herm, sol_classical_herm[0]/sol_classical_herm[1])
    # print('Classical Solution normalized:', sol_classical_herm/np.linalg.norm(sol_classical_herm), np.linalg.norm(sol_classical_herm))
    
    if is_print_figs_result:
        fig, ax = plt.subplots()
        ax.plot(y)
        fig.savefig('HHL_result_Pennylane.png')



    
    
