# local
from ivy.core.container import Container
from ivy_builder.specs.dataset_dirs import DatasetDirs


class DatasetSpec(Container):

    def __init__(self,
                 dirs: DatasetDirs,
                 **kwargs) -> None:
        """
        base class for storing general properties of the dataset which is saved on disk
        """
        super().__init__(kwargs)
        self['dirs'] = dirs
