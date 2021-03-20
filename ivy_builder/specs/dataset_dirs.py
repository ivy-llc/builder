# local
from ivy.core.container import Container


class DatasetDirs(Container):

    def __init__(self,
                 **kwargs) -> None:
        """
        base class for storing directories necessary for the data loader
        """
        super().__init__(kwargs)
