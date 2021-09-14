# global
import importlib

# local
import ivy
from ivy.core.container import Container
from ivy_builder.specs import DatasetSpec


# ToDo: fix cyclic imports, so this method can be imported from the builder module
def load_class_from_str(full_str):
    mod_str = '.'.join(full_str.split('.')[:-1])
    class_str = full_str.split('.')[-1]
    return getattr(importlib.import_module(mod_str), class_str)


class NetworkSpec(Container):

    def __init__(self, dataset_spec: DatasetSpec = None, device: str = 'cpu:0', **kwargs) -> None:
        """
        base class for storing general specifications of the neural network
        """
        super().__init__(kwargs)
        if 'subnets' in self:
            for k, subet_spec in self.subnets.items():
                self.subnets[k].network_class = load_class_from_str(subet_spec.network_class)
                self.subnets[k].store_vars = ivy.default(self.subnets[k].if_exists('store_vars'), True)
                self.subnets[k].dataset_spec = dataset_spec
                self.subnets[k].device = device
        self['dataset_spec'] = dataset_spec
        self['device'] = device
