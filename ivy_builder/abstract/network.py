# global
import ivy

# local
from ivy_builder.specs.network_spec import NetworkSpec


class Network(ivy.Module):

    def __init__(self,
                 network_spec: NetworkSpec,
                 v=None) -> None:
        """
        base class for any networks
        """
        super(Network, self).__init__(v=v, dev_str=network_spec.device)
        self._spec = network_spec

    # Getters #
    # --------#

    @property
    def spec(self):
        return self._spec

    @property
    def device(self):
        return self._spec.device
