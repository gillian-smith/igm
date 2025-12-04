from .mapping import Mapping
from .mapping_network import MappingNetwork
from .mapping_identity import MappingIdentity
from .mapping_data_assimilation import MappingDataAssimilation
from .mapping_combined_data_assimilation import MappingCombinedDataAssimilation

Mappings = {
    "identity": MappingIdentity,
    "network": MappingNetwork,
    "data_assimilation": MappingDataAssimilation,
    "combined_data_assimilation": MappingCombinedDataAssimilation,
}

from .interfaces import InterfaceMapping, InterfaceMappings
