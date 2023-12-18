#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import os
import igm
from typing import List, Any


def run_intializers(modules: List, params: Any, state: igm.State) -> None:
    for module in modules:
        module.initialize(params, state)

def run_processes(modules: List, params: Any, state: igm.State) -> None:
    if hasattr(state, "t"):
        while state.t < params.time_end:
            with tf.profiler.experimental.Trace('process_profile', step_num=state.t, _r=1):
                for module in modules:
                    module.update(params, state)


def run_finalizers(modules: List, params: Any, state: igm.State) -> None:
    for module in modules:
        module.finalize(params, state)


def main():
    print("-----------------------------------------------------------------")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    print("-----------------------------------------------------------------")

    # Collect defaults, overide from json file, and parse all core parameters
    parser = igm.params_core()
    params, __ = parser.parse_known_args()

    modules_dict = igm.get_modules_list(params.param_file)
    imported_modules = igm.load_modules(modules_dict)
    imported_modules = igm.add_dependencies(imported_modules)

    # Collect defaults, overide from json file, and parse all specific module parameters
    for module in imported_modules:
        module.params(parser)


    # igm.overide_from_json_file(parser, check_if_params_exist=True) is this needed?? Does not seem like a good solution
    # if you just want to give a message to the users, you can use parser.error or a try-except catch
    params = parser.parse_args()  # args=[] add this for jupyter notebook
    
    # print definive parameters in a file for record
    if params.print_params:
        igm.print_params(params)

    # Define a state class/dictionnary that contains all the data
    state = igm.State()

    # if logging is activated, add a logger to the state
    if params.logging:
        igm.add_logger(params, state)
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            
    # Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
    with tf.device("/GPU:"+str(params.gpu)):
        # Initialize all the model components in turn
        run_intializers(imported_modules, params, state)
        tf.profiler.experimental.server.start(6009)
        run_processes(imported_modules, params, state)
        tf.profiler.experimental.stop()
        run_finalizers(imported_modules, params, state)




if __name__ == "__main__":
    main()
