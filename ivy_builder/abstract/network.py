# global
import ivy
import abc

# local
from ivy_builder.specs.network_spec import NetworkSpec


class Network(ivy.Module):

    # noinspection PyMissingConstructor
    def __init__(self,
                 network_spec: NetworkSpec,
                 v=None) -> None:
        """
        base class for any networks
        """
        self._spec = network_spec
        ivy.Module.__init__(self, v=v, dev_str=self._spec.device, build_mode=self._spec.build_mode)

    @abc.abstractmethod
    def _build(self) -> None:
        """
        Network builder method, to be overriden
        """
        raise NotImplementedError

    # Getters #
    # --------#

    @property
    def spec(self):
        return self._spec

    @property
    def device(self):
        return self._spec.device
