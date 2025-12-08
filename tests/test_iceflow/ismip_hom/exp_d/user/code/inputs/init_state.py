#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from tests.test_iceflow.ismip_hom.utils import ExperimentD


def run(cfg, state):

    L = float(cfg.inputs.init_state.L)
    n = int(cfg.inputs.init_state.n)
    dtype = tf.float32

    ExperimentD.init_state(state, L, n, dtype)
