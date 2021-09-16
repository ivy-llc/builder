# global
import abc
import copy

# local
from ivy_builder.specs.spec import Spec
from ivy_builder.specs.dataset_dirs import DatasetDirs


class DatasetSpec(Spec, abc.ABC):

    def __init__(self,
                 dirs: DatasetDirs,
                 **kwargs) -> None:
        """
        base class for storing general properties of the dataset which is saved on disk
        """
        kw = copy.deepcopy(locals()['kwargs'])
        super().__init__(dirs=dirs,
                         **kwargs)
        self._kwargs = kw
