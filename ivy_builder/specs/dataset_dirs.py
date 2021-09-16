# global
import abc

# local
from ivy_builder.specs.spec import Spec
from ivy_builder.specs.spec import locals_to_kwargs


class DatasetDirs(Spec, abc.ABC):

    def __init__(self,
                 **kwargs) -> None:
        """
        base class for storing directories necessary for the data loader
        """
        kw = locals_to_kwargs(locals())
        super().__init__(**kwargs)
        self._kwargs = kw
