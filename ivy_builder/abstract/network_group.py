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
        self._subnets = ivy.Container()
        super(NetworkGroup, self).__init__(spec, v=v)

    def _build_subnets(self, *args, **kwargs) -> bool:
        """
        Build the network subnets.
        """
        for k, subnet_spec in self._subnet_specs.items():
            subnet = subnet_spec.network_class(subnet_spec, v=ivy.default(lambda: self._v_in[k], None, True))
            if subnet_spec.build_mode == 'explicit':
                subnet.build(*args, **kwargs, store_vars=ivy.default(subnet_spec.if_exists('store_vars'), True))
            self._subnets[k] = subnet
            setattr(self, k, subnet)
        return bool(np.prod([subnet.built for subnet in self._subnets.values()]))

    def _build(self, *args, **kwargs) -> bool:
        """
        Network builder method. This should be overriden if additional building if required.
        """
        self._build_subnets(*args, **kwargs)
        return bool(np.prod([subnet.built for subnet in self._subnets.values()]))
