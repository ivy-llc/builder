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
        Network builder method, to be overriden
        """
        raise NotImplementedError

    def build(self, store_vars=True):
        self._build()
        ivy.Module.__init__(self, v=self._v_in, dev_str=self._spec.device)
        if not store_vars:
            # ToDo: verify variables in self.v created during ivy.Module.__init__ are released once this method exits
            self.v = ivy.Container()

    # Getters #
    # --------#

    @property
    def spec(self):
        return self._spec

    @property
    def device(self):
        return self._spec.device
