# global
import ivy
import abc
import importlib
from typing import List

# local
from ivy_builder.specs.spec import Spec
from ivy_builder.specs import DatasetSpec
from ivy_builder.specs.spec import locals_to_kwargs


# ToDo: fix cyclic imports, so this method can be imported from the builder module
def load_class_from_str(full_str):
    mod_str = '.'.join(full_str.split('.')[:-1])
    class_str = full_str.split('.')[-1]
    return getattr(importlib.import_module(mod_str), class_str)


class NetworkSpec(Spec, abc.ABC):

    def __init__(self, dataset_spec: DatasetSpec = None, dev_strs: List[str] = None,
                 v_keychains=None, keep_v_keychains=False, build_mode='explicit', **kwargs) -> None:
        """
        base class for storing general specifications of the neural network
        """
        kw = locals_to_kwargs(locals())
        super().__init__(dataset_spec=dataset_spec,
                         dev_strs=dev_strs,
                         v_keychains=v_keychains,
                         keep_v_keychains=keep_v_keychains,
                         build_mode=build_mode,
                         **kwargs)
        if 'subnets' in self:
            for k, subet_spec in self.subnets.items():
                if 'network_spec_class' in subet_spec:
                    if isinstance(subet_spec.network_spec_class, str):
                        spec_class = load_class_from_str(subet_spec.network_spec_class)
                    else:
                        spec_class = subet_spec.network_spec_class
                    if isinstance(kwargs['subnets'][k], spec_class):
                        subet_spec = kwargs['subnets'][k]
                    else:
                        subet_spec = spec_class(**{**kwargs['subnets'][k],
                                                   **dict(dataset_spec=dataset_spec, dev_strs=dev_strs)})
                    self.subnets[k] = subet_spec
                if isinstance(subet_spec.network_class, str):
                    self.subnets[k].network_class = load_class_from_str(subet_spec.network_class)
                else:
                    self.subnets[k].network_class = subet_spec.network_class
                self.subnets[k].store_vars = ivy.default(self.subnets[k].if_exists('store_vars'), True)
                self.subnets[k].build_mode = ivy.default(self.subnets[k].if_exists('build_mode'), self.build_mode)
                self.subnets[k].dataset_spec = dataset_spec
                self.subnets[k].dev_strs = dev_strs
        self._kwargs = kw
