# local
from ivy.core.container import Container
from ivy_builder.abstract.network import Network
from ivy_builder.specs.dataset_spec import DatasetSpec


class DataLoaderSpec(Container):

    def __init__(self,
                 dataset_spec: DatasetSpec,
                 **kwargs) -> None:
        """
        base class for storing general parameters which define the way in which the data loader loads the dataset
        """
        super().__init__(kwargs)
        self['dataset_spec'] = dataset_spec
