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
        self._dev_str = ivy.default(lambda: network_spec.dev_strs[0], ivy.default_device(), True)
        ivy.Module.__init__(self, v=v, dev_str=self._spec.dev_strs[0],
                            build_mode=ivy.default(self._spec.if_exists('build_mode'), 'explicit'),
                            store_vars=ivy.default(self._spec.if_exists('store_vars'), True))

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
