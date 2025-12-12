from .interface import InterfaceMapping
from .network import InterfaceNetwork
from .identity import InterfaceIdentity
from .data_assimilation import InterfaceDataAssimilation
from .combined_data_assimilation import InterfaceCombinedDataAssimilation

InterfaceMappings = {
    "identity": InterfaceIdentity,
    "network": InterfaceNetwork,
    "data_assimilation": InterfaceDataAssimilation,
    "combined_data_assimilation": InterfaceCombinedDataAssimilation,
}
