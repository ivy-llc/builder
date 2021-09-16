# global
import abc

# local
from ivy_builder.specs.spec import Spec


class DatasetDirs(Spec, abc.ABC):

    def __init__(self,
                 **kwargs) -> None:
        """
        base class for storing directories necessary for the data loader
        """
        self._locals = locals()
        super().__init__(**kwargs)
