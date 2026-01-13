import warnings
from typing import Any, Dict
from omegaconf import DictConfig
from .interface import InterfaceMapping
from igm.common import State
from igm.processes.iceflow.emulate.utils.artifacts import load_emulator_artifact
from igm.processes.iceflow.emulate.utils.architectures import Architectures
from igm.processes.iceflow.emulate import Architectures, NormalizationsDict
from igm.processes.iceflow.unified.mappings import Mappings
from igm.processes.iceflow.unified.bcs.utils import init_bcs


class InterfaceNetwork(InterfaceMapping):

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_physics = cfg.processes.iceflow.physics
        cfg_unified = cfg.processes.iceflow.unified

        inputs = list(cfg_unified.inputs)
        Nz = int(cfg_numerics.Nz)

        if cfg_unified.network.pretrained:
            artifact_dir = cfg_unified.network.pretrained_path
            # print path
            print(f"Loading pretrained model from: {artifact_dir}")
            iceflow_model, _manifest = load_emulator_artifact(artifact_dir, cfg)

        else:
            warnings.warn("No pretrained emulator selected. Starting from scratch.")

            # Compute nb_inputs consistently
            nb_inputs = len(inputs) + (cfg_physics.dim_arrhenius == 3) * (Nz - 1)
            nb_outputs = 2 * Nz

            arch_name = cfg_unified.network.architecture
            if arch_name not in Architectures:
                raise ValueError(f"Unknown network architecture: {arch_name}. Available: {Architectures.keys()}")

            iceflow_model = Architectures[arch_name](cfg, nb_inputs, nb_outputs)

            # Build normalizer and attach to model (single source of truth)
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

        # NOTE: normalizer removed from mapping args
        return {
            "bcs": bcs,
            "network": state.iceflow_model,
            "Nz": cfg_numerics.Nz,
            "output_scale": cfg_unified.network.output_scale,
            "precision": cfg_numerics.precision,
        }
