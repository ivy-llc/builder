# local
from ivy.core.container import Container
from ivy_builder.specs import DatasetSpec


class NetworkSpec(Container):

    def __init__(self, dataset_spec: DatasetSpec = None, device: str = 'cpu:0', **kwargs) -> None:
        """
        base class for storing general specifications of the neural network
        """
        super().__init__(kwargs)
        self['dataset_spec'] = dataset_spec
        self['device'] = device
