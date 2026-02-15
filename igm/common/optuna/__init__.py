# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Optuna integration for IGM (Tier 2: Hydra sweeper plugin)."""

from .register import register_sweeper

register_sweeper()
