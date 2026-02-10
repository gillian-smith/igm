import tensorflow as tf

import warnings
from typing import Any, Dict
from omegaconf import DictConfig
from .interface import InterfaceMapping
from igm.common import State
from igm.processes.iceflow.emulate.utils.artifacts import load_emulator_artifact
from igm.processes.iceflow.emulate.utils.architectures import Architectures
from igm.processes.iceflow.emulate.utils import NormalizationsDict
from igm.processes.iceflow.unified.mappings import Mappings
from igm.processes.iceflow.unified.bcs.utils import init_bcs
from igm.utils.math.precision import normalize_precision

class InterfaceNetwork(InterfaceMapping):

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_physics = cfg.processes.iceflow.physics
        cfg_unified = cfg.processes.iceflow.unified

        if cfg.processes.iceflow.do_pretraining:
            inputs = list(cfg.processes.pretraining.inputs)
        else:
            inputs = list(cfg_unified.inputs)
            
        Nz = int(cfg_numerics.Nz)

        if cfg_unified.network.pretrained:
            dtype = normalize_precision(cfg_numerics.precision)
            artifact_dir = cfg_unified.network.pretrained_path
            tf.keras.mixed_precision.set_global_policy("float64" if tf.as_dtype(dtype) == tf.float64 else "float32")
            iceflow_model, _manifest = load_emulator_artifact(artifact_dir, cfg)
        else:
            warnings.warn("No pretrained emulator selected. Starting from scratch.")

            nb_inputs = len(inputs) + (cfg_physics.dim_arrhenius == 3) * (Nz - 1)
            nb_outputs = 2 * Nz

            arch_name = cfg_unified.network.architecture
            if arch_name not in Architectures:
                raise ValueError(f"Unknown network architecture: {arch_name}. Available: {Architectures.keys()}")

            iceflow_model = Architectures[arch_name](cfg, nb_inputs, nb_outputs)

            # Build normalizer and attach to model
            if cfg.processes.iceflow.do_pretraining:
                iceflow_model.input_normalizer = None # this is handled in pretraining process
            else:
                # Inference / non-pretraining: keep Brandon's config-driven behavior for now
                method = cfg_unified.normalization.method
                normalizing_class = NormalizationsDict[method]

                if method == "adaptive":
                    normalizing_layer = normalizing_class(nb_inputs)
                elif method == "fixed":
                    offsets = cfg_unified.normalization.fixed.inputs_offsets
                    variances = cfg_unified.normalization.fixed.inputs_variances
                    normalizing_layer = normalizing_class(offsets, variances)
                elif method in ("automatic", "none"):
                    normalizing_layer = normalizing_class()
                else:
                    raise ValueError(f"Unknown normalizing method: {method}")

                iceflow_model.input_normalizer = normalizing_layer


        state.iceflow_model = iceflow_model
        state.iceflow_model.compile(jit_compile=False)

        bcs = init_bcs(cfg, state, cfg_unified.bcs)

        return {
            "bcs": bcs,
            "network": state.iceflow_model,
            "Nz": cfg_numerics.Nz,
            "output_scale": cfg_unified.network.output_scale,
            "precision": cfg_numerics.precision,
        }
