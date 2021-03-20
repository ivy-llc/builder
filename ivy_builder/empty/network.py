# local
from ivy_builder.specs import NetworkSpec
from ivy_builder.abstract.network import Network


class EmptyNetwork(Network):

    def __init__(self,
                 network_spec: NetworkSpec) -> None:
        """
        base class for any networks to be used with this core tensorflow training repo
        """
        super(EmptyNetwork, self).__init__(network_spec)

    def call(self, x):
        """
        call a single forward pass of the network
        """
        pass

    def get_serializable_model(self, input_shape):
        pass
