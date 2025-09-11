# hydra_optuna_path_callback.py
from hydra.core.hydra_config import HydraConfig
from hydra.experimental.callback import Callback

try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    _OPTUNA_AVAILABLE = False


class OptunaPathCallback(Callback):
    def on_run_start(self, config, **kwargs):
        if not _OPTUNA_AVAILABLE:
            return  # silently skip if Optuna not installed

        output_dir = HydraConfig.get().runtime.output_dir
        trial = getattr(config, "_optuna_trial", None)
        if trial is not None and isinstance(trial, optuna.trial.Trial):
            trial.set_user_attr("hydra_output_dir", output_dir)