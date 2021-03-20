# local
from ivy.core.container import Container


class NetworkSpec(Container):

    def __init__(self, **kwargs) -> None:
        """
        base class for storing general specifications of the neural network
        """
        super().__init__(kwargs)
