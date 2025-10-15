#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from abc import ABC, abstractmethod


class EnergyComponent(ABC):
    """Generic energy component."""

    @abstractmethod
    def cost():
        """Abstract cost computation."""
        pass
