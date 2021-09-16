# global
import abc
import copy

# local
from ivy_builder.specs.spec import Spec


class DatasetDirs(Spec, abc.ABC):

    def __init__(self,
                 **kwargs) -> None:
        """
        base class for storing directories necessary for the data loader
        """
        kw = copy.deepcopy(locals()['kwargs'])
        super().__init__(**kwargs)
        self._kwargs = kw
