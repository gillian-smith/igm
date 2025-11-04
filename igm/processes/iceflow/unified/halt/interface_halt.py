#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
from typing import List, Dict, Any

from .criteria import Criteria, Criterion
from .metrics import Metrics


class InterfaceHalt:

    @staticmethod
    def get_halt_args(cfg: DictConfig) -> Dict[str, Any]:
        cfg_halt = cfg.processes.iceflow.unified.halt
        cfg_numerics = cfg.processes.iceflow.numerics

        crit_success = InterfaceHalt._create_crit_list(
            cfg_halt.success, cfg_halt, cfg_numerics
        )
        crit_failure = InterfaceHalt._create_crit_list(
            cfg_halt.failure, cfg_halt, cfg_numerics
        )

        return {
            "crit_success": crit_success,
            "crit_failure": crit_failure,
            "freq": cfg_halt.freq,
        }

    @staticmethod
    def _create_crit_list(
        crit_cfg: DictConfig, cfg_halt: DictConfig, cfg_numerics: DictConfig
    ) -> List[Criterion]:
        crit_names = crit_cfg.criteria
        metric_names = crit_cfg.metrics

        if len(crit_names) != len(metric_names):
            raise ValueError(
                f"❌ Number of criteria ({len(crit_names)}) must match "
                f"number of metrics ({len(metric_names)})."
            )

        crit_list = []
        for i, (crit_name, metric_name) in enumerate(zip(crit_names, metric_names)):
            metric_name_override = (
                f"{metric_name}_{i+1}"
                if metric_names.count(metric_name) > 1
                else metric_name
            )
            metric_args = InterfaceHalt._get_metric_args(
                default_name=metric_name,
                override_name=metric_name_override,
                default_cfg=cfg_halt.metrics,
                override_cfg=crit_cfg,
            )
            metric_args = {"dtype": cfg_numerics.precision, **metric_args}
            metric_class = Metrics[metric_name]
            metric = metric_class(**metric_args)

            crit_name_override = (
                f"{crit_name}_{i+1}" if crit_names.count(crit_name) > 1 else crit_name
            )
            crit_args = InterfaceHalt._get_crit_args(
                default_name=crit_name,
                override_name=crit_name_override,
                default_cfg=cfg_halt.criteria,
                override_cfg=crit_cfg,
            )
            crit_class = Criteria[crit_name]
            crit = crit_class(metric=metric, **crit_args)
            crit_list.append(crit)

        return crit_list

    @staticmethod
    def _get_metric_args(
        default_name: str,
        override_name: str,
        default_cfg: DictConfig,
        override_cfg: DictConfig,
    ) -> Dict[str, Any]:
        default_args = {}
        if default_name in default_cfg:
            default_args = dict(default_cfg[default_name].items())

        override_args = {}
        for name in (override_name, default_name):
            if hasattr(override_cfg, name):
                override_args = dict(getattr(override_cfg, name).items())
                break

        return {**default_args, **override_args}

    @staticmethod
    def _get_crit_args(
        default_name: str,
        override_name: str,
        default_cfg: DictConfig,
        override_cfg: DictConfig,
    ) -> Dict[str, Any]:
        default_args = {}
        if default_name in default_cfg:
            default_args = dict(default_cfg[default_name].items())

        override_args = {}
        for name in (override_name, default_name):
            if hasattr(override_cfg, name):
                override_args = dict(getattr(override_cfg, name).items())
                break

        return {**default_args, **override_args}
