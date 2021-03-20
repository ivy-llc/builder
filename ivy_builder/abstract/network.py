# global
import abc

# local
from ivy_builder.specs.network_spec import NetworkSpec


class Network(abc.ABC):

    def __init__(self,
                 network_spec: NetworkSpec) -> None:
        """
        base class for any networks
        """
        super(Network, self).__init__()
        self._spec = network_spec

    def get_serializable_model(self):
        raise Exception('Network has not implemented get_serializable_model().\n'
                        'Therefore, there is no model available for saving in SavedModel format.')

    def test_serializable_model(self, model_in):
        raise Exception('Network has not implemented set_serializable_model().\n'
                        'Therefore, models saved in SavedModel format cannot be tested.')

    # Getters #
    # --------#

    @property
    def spec(self):
        return self._spec
