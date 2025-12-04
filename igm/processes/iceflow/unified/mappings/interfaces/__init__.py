from .interface import InterfaceMapping
from .interface_network import InterfaceNetwork
from .interface_identity import InterfaceIdentity
from .interface_data_assimilation import InterfaceDataAssimilation
from .interface_combined_data_assimilation import InterfaceCombinedDataAssimilation

InterfaceMappings = {
    "identity": InterfaceIdentity,
    "network": InterfaceNetwork,
    "data_assimilation": InterfaceDataAssimilation,
    "combined_data_assimilation": InterfaceCombinedDataAssimilation,
}
