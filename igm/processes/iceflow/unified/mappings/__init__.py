from .mapping import Mapping
from .network import MappingNetwork
from .identity import MappingIdentity
from .data_assimilation import MappingDataAssimilation
from .combined_data_assimilation import MappingCombinedDataAssimilation

Mappings = {
    "identity": MappingIdentity,
    "network": MappingNetwork,
    "data_assimilation": MappingDataAssimilation,
    "combined_data_assimilation": MappingCombinedDataAssimilation,
}

from .interfaces import InterfaceMapping, InterfaceMappings
