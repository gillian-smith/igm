import numpy as np
import matplotlib.pyplot as plt
import os

def params(parser):
    pass

def initialize(params, state):
    if len(params.iflo_save_cost_emulator)>0:
        plot_cost_emulator(params,state)

def update(params, state):
    pass

def finalize(params, state):
    pass

def plot_cost_emulator(params,state):
    file_path = params.iflo_output_directory+"/"+params.iflo_save_cost_emulator+'-'+str(state.it)+'.dat'
    with open(file_path) as file:
        lines = file.readlines()
        costs = [np.array(line.strip().split(), dtype=float) for line in lines]
                 
    costs = np.stack(costs)
    iters = np.arange(costs.shape[0])

    fig, ax = plt.subplots(1,1,figsize=(10, 10))
    ax.plot(iters,costs)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Emulator cost')
    ax.set_yscale('symlog')

    fig.tight_layout()

    save_path = params.iflo_output_directory+"/"+params.iflo_save_cost_emulator+'-'+str(state.it)+'.png'
    fig.savefig(save_path, pad_inches=0)
    plt.close("all")

    os.system(
        "echo rm "
        + save_path
        + " >> clean.sh"
    )