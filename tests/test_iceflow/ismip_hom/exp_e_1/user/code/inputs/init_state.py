#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from tests.test_iceflow.ismip_hom.utils import ExperimentE1


def run(cfg, state):

    dtype = tf.float32

    ExperimentE1.init_state(state, dtype=dtype)
