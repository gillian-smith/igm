from .mapping import Mapping
from .mapping_network import MappingNetwork
from .mapping_identity import MappingIdentity
from .interface_network import InterfaceNetwork
from .interface_identity import InterfaceIdentity
from .mapping_data_assimilation import MappingDataAssimilation
from .interface_data_assimilation import InterfaceDataAssimilation
from .mapping_combined_data_assimilation import MappingCombinedDataAssimilation
from .interface_combined_data_assimilation import InterfaceCombinedDataAssimilation

Mappings = {
    "identity": MappingIdentity,
    "network": MappingNetwork,
    "data_assimilation": MappingDataAssimilation,
    "combined_data_assimilation": MappingCombinedDataAssimilation,
}

InterfaceMappings = {
    "identity": InterfaceIdentity,
    "network": InterfaceNetwork,
    "data_assimilation": InterfaceDataAssimilation,
    "combined_data_assimilation": InterfaceCombinedDataAssimilation,
}