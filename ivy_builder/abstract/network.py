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
        self._v_in = v
        self._spec = network_spec

    @abc.abstractmethod
    def _build(self) -> None:
        """
        compute learning rate, given global step
        """
        raise NotImplementedError

    def build(self):
        self._build()
        ivy.Module.__init__(self, v=self._v_in, dev_str=self._spec.device)

    # Getters #
    # --------#

    @property
    def spec(self):
        return self._spec

    @property
    def device(self):
        return self._spec.device
