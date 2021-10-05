# global
import abc
import ivy
from typing import Union, List

# local
from ivy_builder.specs.spec import Spec
from ivy_builder.specs.spec import locals_to_kwargs
from ivy_builder.specs.dataset_spec import DatasetSpec


class DataLoaderSpec(Spec, abc.ABC):

    def __init__(self,
                 dataset_spec: DatasetSpec,
                 dev_strs: Union[str, List[str]] = None,
                 **kwargs) -> None:
        """
        base class for storing general parameters which define the way in which the data loader loads the dataset
        """
        kw = locals_to_kwargs(locals())
        super().__init__(dataset_spec=dataset_spec,
                         dev_strs=ivy.default(dev_strs, ['gpu:0'] if ivy.gpu_is_available() else ['cpu']),
                         **kwargs)
        self._kwargs = kw
