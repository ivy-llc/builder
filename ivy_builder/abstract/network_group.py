# global
import ivy
from abc import ABC

# local
from ivy_builder.specs.network_spec import NetworkSpec
from ivy_builder.abstract.network import Network as BaseNetwork


class NetworkGroup(BaseNetwork, ABC):

    # noinspection PyMissingConstructor
    def __init__(self,
                 spec: NetworkSpec,
                 v=None) -> None:
        """
        base class for any networks
        """
        self._v_in = v
        self._spec = spec
        self._subnet_specs = spec.subnets
        super(NetworkGroup, self).__init__(spec, v=v)

    def _build(self) -> None:
        """
        Network builder method.
        """
        for k, subnet_spec in self._subnet_specs.items():
            subnet = subnet_spec.network_class(subnet_spec, v=ivy.default(lambda: self._v_in[k], None, True))
            subnet.build(store_vars=ivy.default(subnet_spec.if_exists('store_vars'), True))
            setattr(self, k, subnet)
