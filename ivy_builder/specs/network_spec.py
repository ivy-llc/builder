# local
from ivy.core.container import Container


class NetworkSpec(Container):

    def __init__(self, device: str = 'cpu:0', **kwargs) -> None:
        """
        base class for storing general specifications of the neural network
        """
        super().__init__(kwargs)
        self['device'] = device
